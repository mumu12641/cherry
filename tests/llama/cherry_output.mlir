// Original IR loaded from file
module {
  cherry.func private @llama_forward(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg3: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg4: !cherry.cherry_tensor<[1x1024xf32]>, %arg5: !cherry.cherry_tensor<[32000x768xf32]>, %arg6: !cherry.cherry_tensor<[12x768xf32]>, %arg7: !cherry.cherry_tensor<[12x768x768xf32]>, %arg8: !cherry.cherry_tensor<[12x768x768xf32]>, %arg9: !cherry.cherry_tensor<[12x768x768xf32]>, %arg10: !cherry.cherry_tensor<[12x768x768xf32]>, %arg11: !cherry.cherry_tensor<[12x768xf32]>, %arg12: !cherry.cherry_tensor<[12x768x2048xf32]>, %arg13: !cherry.cherry_tensor<[12x2048x768xf32]>, %arg14: !cherry.cherry_tensor<[12x768x2048xf32]>, %arg15: !cherry.cherry_tensor<[768xf32]>, %arg16: !cherry.cherry_tensor<[32000x768xf32]>) -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.tensor_slice %arg5[%arg0, %0] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = cherry.constant(1 : i64) : i64
    %c12 = arith.constant 12 : index
    %3 = cherry.constant(768 : i64) : i64
    %4 = cherry.constant(2048 : i64) : i64
    %c12_0 = arith.constant 12 : index
    %5 = cherry.constant(12 : i64) : i64
    %6 = cherry.constant(64 : i64) : i64
    %7 = cherry.constant(1.250000e-01 : f32) : f32
    %8 = cherry.constant(1024 : i64) : i64
    %9 = cherry.constant(32000 : i64) : i64
    %10:3 = scf.for %arg17 = %c0 to %c12 step %c1 iter_args(%arg18 = %1, %arg19 = %arg2, %arg20 = %arg3) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
      %14 = arith.index_cast %arg17 : index to i64
      %15 = cherry.tensor_slice %arg6[%14, %0] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %16 = cherry.reshape %15, %3 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
      %17 = cherry.rmsnorm %arg18 scale %16 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
      %18 = cherry.tensor_slice %arg7[%14, %0, %0] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
      %19 = cherry.reshape %18, %3, %3 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
      %20 = cherry.tensor_slice %arg8[%14, %0, %0] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
      %21 = cherry.reshape %20, %3, %3 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
      %22 = cherry.tensor_slice %arg9[%14, %0, %0] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
      %23 = cherry.reshape %22, %3, %3 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
      %24 = cherry.matmul %17, %19 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %25 = cherry.matmul %17, %21 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %26 = cherry.matmul %17, %23 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %27 = cherry.reshape %24, %2, %5, %6 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
      %28 = cherry.rope %27, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
      %29 = cherry.reshape %28, %2, %3 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %30 = cherry.reshape %25, %2, %5, %6 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
      %31 = cherry.rope %30, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
      %32 = cherry.reshape %31, %2, %3 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %33 = cherry.reshape %26, %2, %5, %6 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
      %34 = cherry.rope %33, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
      %35 = cherry.reshape %34, %2, %3 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %36 = cherry.reshape %32, %2, %2, %3 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
      %37 = cherry.tensor_set_slice %arg19[%14, %arg1], %36 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
      %38 = cherry.reshape %35, %2, %2, %3 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
      %39 = cherry.tensor_set_slice %arg20[%14, %arg1], %38 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
      %40 = cherry.tensor_slice %37[%14, %0, %0] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
      %41 = cherry.reshape %40, %8, %3 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
      %42 = cherry.tensor_slice %39[%14, %0, %0] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
      %43 = cherry.reshape %42, %8, %3 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
      %44 = cherry.create_tensor dense<0.000000e+00> : tensor<1x12x64xf32> -> !cherry.cherry_tensor<[1x12x64xf32]>
      %45 = scf.for %arg21 = %c0 to %c12_0 step %c1 iter_args(%arg22 = %44) -> (!cherry.cherry_tensor<[1x12x64xf32]>) {
        %66 = arith.index_cast %arg21 : index to i64
        %67 = arith.muli %66, %6 : i64
        %68 = cherry.tensor_slice %29[%0, %67] sizes [1, 64] : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x64xf32]>
        %69 = cherry.tensor_slice %41[%0, %67] sizes [1024, 64] : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
        %70 = cherry.transpose %69 perm [1, 0] : (!cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[64x1024xf32]>
        %71 = cherry.masked_matmul %68, %70, %arg4 : (!cherry.cherry_tensor<[1x64xf32]>, !cherry.cherry_tensor<[64x1024xf32]>, !cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
        %72 = cherry.tensor_mul_scalar %71, %7 : (!cherry.cherry_tensor<[1x1024xf32]>, f32) -> !cherry.cherry_tensor<[1x1024xf32]>
        %73 = cherry.softmax %72 axis 1 : (!cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
        %74 = cherry.tensor_slice %43[%0, %67] sizes [1024, 64] : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
        %75 = cherry.matmul %73, %74 : (!cherry.cherry_tensor<[1x1024xf32]>, !cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[1x64xf32]>
        %76 = cherry.reshape %75, %2, %2, %6 : (!cherry.cherry_tensor<[1x64xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x64xf32]>
        %77 = cherry.tensor_set_slice %arg22[%0, %66], %76 : (!cherry.cherry_tensor<[1x12x64xf32]>, !cherry.cherry_tensor<[1x1x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        scf.yield %77 : !cherry.cherry_tensor<[1x12x64xf32]>
      }
      %46 = cherry.reshape %45, %2, %3 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %47 = cherry.tensor_slice %arg10[%14, %0, %0] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
      %48 = cherry.reshape %47, %3, %3 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
      %49 = cherry.matmul %46, %48 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %50 = cherry.tensor_add %arg18, %49 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %51 = cherry.tensor_slice %arg11[%14, %0] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %52 = cherry.reshape %51, %3 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
      %53 = cherry.rmsnorm %50 scale %52 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
      %54 = cherry.tensor_slice %arg12[%14, %0, %0] sizes [1, 768, 2048] : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
      %55 = cherry.reshape %54, %3, %4 : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
      %56 = cherry.tensor_slice %arg14[%14, %0, %0] sizes [1, 768, 2048] : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
      %57 = cherry.reshape %56, %3, %4 : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
      %58 = cherry.matmul %53, %55 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
      %59 = cherry.matmul %53, %57 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
      %60 = cherry.tensor_silu %58 : (!cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
      %61 = cherry.tensor_mul %60, %59 : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
      %62 = cherry.tensor_slice %arg13[%14, %0, %0] sizes [1, 2048, 768] : (!cherry.cherry_tensor<[12x2048x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x2048x768xf32]>
      %63 = cherry.reshape %62, %4, %3 : (!cherry.cherry_tensor<[1x2048x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[2048x768xf32]>
      %64 = cherry.matmul %61, %63 : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[2048x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %65 = cherry.tensor_add %50, %64 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      scf.yield %65, %37, %39 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    }
    %11 = cherry.rmsnorm %10#0 scale %arg15 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
    %12 = cherry.reshape %arg16, %3, %9 : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
    %13 = cherry.matmul %11, %12 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
    cherry.return %13, %10#1, %10#2 : !cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
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
    %10 = cherry.create_tensor dense<1.200000e+01> : tensor<768xf32> -> !cherry.cherry_tensor<[768xf32]>
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
      %18 = cherry.generate_mask %arg1, [1, 1024] : !cherry.cherry_tensor<[1x1024xf32]>
      %19:3 = cherry.call @llama_forward(%arg0, %arg1, %arg2, %arg3, %18, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11) : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1024xf32]>, !cherry.cherry_tensor<[32000x768xf32]>, !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[768xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>)
      %20 = cherry.argmax %19#0 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      cherry.print %19#0 : !cherry.cherry_tensor<[1x32000xf32]>
      %21 = cherry.constant(0 : i64) : i64
      %22 = cherry.tensor_get %20[%21] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %23 = cherry.constant(1 : i64) : i64
      %24 = arith.addi %arg1, %23 : i64
      scf.yield %22, %24, %19#1, %19#2 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
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
    %5 = cherry.create_tensor dense<7.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %6 = cherry.create_tensor dense<8.000000e+00> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %7 = cherry.create_tensor dense<9.000000e+00> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %8 = cherry.create_tensor dense<1.000000e+01> : tensor<12x2048x768xf32> -> !cherry.cherry_tensor<[12x2048x768xf32]>
    %9 = cherry.create_tensor dense<1.100000e+01> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %10 = cherry.create_tensor dense<1.200000e+01> : tensor<768xf32> -> !cherry.cherry_tensor<[768xf32]>
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
      %18 = cherry.generate_mask %arg1, [1, 1024] : !cherry.cherry_tensor<[1x1024xf32]>
      %19 = cherry.constant(0 : i64) : i64
      %20 = cherry.tensor_slice %0[%arg0, %19] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %21 = cherry.constant(1 : i64) : i64
      %22 = cherry.constant(768 : i64) : i64
      %23 = cherry.constant(2048 : i64) : i64
      %24 = cherry.constant(12 : i64) : i64
      %25 = cherry.constant(64 : i64) : i64
      %26 = cherry.constant(1.250000e-01 : f32) : f32
      %27 = cherry.constant(1024 : i64) : i64
      %28 = cherry.constant(32000 : i64) : i64
      %29:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %20, %arg6 = %arg2, %arg7 = %arg3) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
        %38 = arith.index_cast %arg4 : index to i64
        %39 = cherry.tensor_slice %1[%38, %19] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %40 = cherry.reshape %39, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %41 = cherry.rmsnorm %arg5 scale %40 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %42 = cherry.tensor_slice %2[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %43 = cherry.reshape %42, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %44 = cherry.tensor_slice %3[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %45 = cherry.reshape %44, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %46 = cherry.tensor_slice %4[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %47 = cherry.reshape %46, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %48 = cherry.matmul %41, %43 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %49 = cherry.matmul %41, %45 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %50 = cherry.matmul %41, %47 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %51 = cherry.reshape %48, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %52 = cherry.rope %51, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %53 = cherry.reshape %52, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %54 = cherry.reshape %49, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %55 = cherry.rope %54, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %56 = cherry.reshape %55, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %57 = cherry.reshape %50, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %58 = cherry.rope %57, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %59 = cherry.reshape %58, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %60 = cherry.reshape %56, %21, %21, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %61 = cherry.tensor_set_slice %arg6[%38, %arg1], %60 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %62 = cherry.reshape %59, %21, %21, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %63 = cherry.tensor_set_slice %arg7[%38, %arg1], %62 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %64 = cherry.tensor_slice %61[%38, %19, %19] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
        %65 = cherry.reshape %64, %27, %22 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
        %66 = cherry.tensor_slice %63[%38, %19, %19] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
        %67 = cherry.reshape %66, %27, %22 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
        %68 = cherry.create_tensor dense<0.000000e+00> : tensor<1x12x64xf32> -> !cherry.cherry_tensor<[1x12x64xf32]>
        %69 = scf.for %arg8 = %c0 to %c12 step %c1 iter_args(%arg9 = %68) -> (!cherry.cherry_tensor<[1x12x64xf32]>) {
          %90 = arith.index_cast %arg8 : index to i64
          %91 = arith.muli %90, %25 : i64
          %92 = cherry.tensor_slice %53[%19, %91] sizes [1, 64] : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x64xf32]>
          %93 = cherry.tensor_slice %65[%19, %91] sizes [1024, 64] : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
          %94 = cherry.transpose %93 perm [1, 0] : (!cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[64x1024xf32]>
          %95 = cherry.masked_matmul %92, %94, %18 : (!cherry.cherry_tensor<[1x64xf32]>, !cherry.cherry_tensor<[64x1024xf32]>, !cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
          %96 = cherry.tensor_mul_scalar %95, %26 : (!cherry.cherry_tensor<[1x1024xf32]>, f32) -> !cherry.cherry_tensor<[1x1024xf32]>
          %97 = cherry.softmax %96 axis 1 : (!cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
          %98 = cherry.tensor_slice %67[%19, %91] sizes [1024, 64] : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
          %99 = cherry.matmul %97, %98 : (!cherry.cherry_tensor<[1x1024xf32]>, !cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[1x64xf32]>
          %100 = cherry.reshape %99, %21, %21, %25 : (!cherry.cherry_tensor<[1x64xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x64xf32]>
          %101 = cherry.tensor_set_slice %arg9[%19, %90], %100 : (!cherry.cherry_tensor<[1x12x64xf32]>, !cherry.cherry_tensor<[1x1x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
          scf.yield %101 : !cherry.cherry_tensor<[1x12x64xf32]>
        }
        %70 = cherry.reshape %69, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %71 = cherry.tensor_slice %5[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %72 = cherry.reshape %71, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %73 = cherry.matmul %70, %72 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %74 = cherry.tensor_add %arg5, %73 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %75 = cherry.tensor_slice %6[%38, %19] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %76 = cherry.reshape %75, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %77 = cherry.rmsnorm %74 scale %76 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %78 = cherry.tensor_slice %7[%38, %19, %19] sizes [1, 768, 2048] : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %79 = cherry.reshape %78, %22, %23 : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
        %80 = cherry.tensor_slice %9[%38, %19, %19] sizes [1, 768, 2048] : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %81 = cherry.reshape %80, %22, %23 : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
        %82 = cherry.matmul %77, %79 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %83 = cherry.matmul %77, %81 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %84 = cherry.tensor_silu %82 : (!cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %85 = cherry.tensor_mul %84, %83 : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %86 = cherry.tensor_slice %8[%38, %19, %19] sizes [1, 2048, 768] : (!cherry.cherry_tensor<[12x2048x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x2048x768xf32]>
        %87 = cherry.reshape %86, %23, %22 : (!cherry.cherry_tensor<[1x2048x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[2048x768xf32]>
        %88 = cherry.matmul %85, %87 : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[2048x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %89 = cherry.tensor_add %74, %88 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        scf.yield %89, %61, %63 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
      }
      %30 = cherry.rmsnorm %29#0 scale %10 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
      %31 = cherry.reshape %11, %22, %28 : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
      %32 = cherry.matmul %30, %31 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
      %33 = cherry.argmax %32 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      cherry.print %32 : !cherry.cherry_tensor<[1x32000xf32]>
      %34 = cherry.constant(0 : i64) : i64
      %35 = cherry.tensor_get %33[%34] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %36 = cherry.constant(1 : i64) : i64
      %37 = arith.addi %arg1, %36 : i64
      scf.yield %35, %37, %29#1, %29#2 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
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
    %5 = cherry.create_tensor dense<7.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %6 = cherry.create_tensor dense<8.000000e+00> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %7 = cherry.create_tensor dense<9.000000e+00> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %8 = cherry.create_tensor dense<1.000000e+01> : tensor<12x2048x768xf32> -> !cherry.cherry_tensor<[12x2048x768xf32]>
    %9 = cherry.create_tensor dense<1.100000e+01> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %10 = cherry.create_tensor dense<1.200000e+01> : tensor<768xf32> -> !cherry.cherry_tensor<[768xf32]>
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
      %18 = cherry.generate_mask %arg1, [1, 1024] : !cherry.cherry_tensor<[1x1024xf32]>
      %19 = cherry.constant(0 : i64) : i64
      %20 = cherry.tensor_slice %0[%arg0, %19] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %21 = cherry.constant(1 : i64) : i64
      %22 = cherry.constant(768 : i64) : i64
      %23 = cherry.constant(2048 : i64) : i64
      %24 = cherry.constant(12 : i64) : i64
      %25 = cherry.constant(64 : i64) : i64
      %26 = cherry.constant(1.250000e-01 : f32) : f32
      %27 = cherry.constant(1024 : i64) : i64
      %28 = cherry.constant(32000 : i64) : i64
      %29:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %20, %arg6 = %arg2, %arg7 = %arg3) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
        %38 = arith.index_cast %arg4 : index to i64
        %39 = cherry.tensor_slice %1[%38, %19] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %40 = cherry.reshape %39, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %41 = cherry.rmsnorm %arg5 scale %40 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %42 = cherry.tensor_slice %2[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %43 = cherry.reshape %42, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %44 = cherry.tensor_slice %3[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %45 = cherry.reshape %44, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %46 = cherry.tensor_slice %4[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %47 = cherry.reshape %46, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %48 = cherry.matmul %41, %43 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %49 = cherry.matmul %41, %45 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %50 = cherry.matmul %41, %47 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %51 = cherry.reshape %48, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %52 = cherry.rope %51, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %53 = cherry.reshape %52, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %54 = cherry.reshape %49, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %55 = cherry.rope %54, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %56 = cherry.reshape %55, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %57 = cherry.reshape %50, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %58 = cherry.rope %57, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %59 = cherry.reshape %58, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %60 = cherry.reshape %56, %21, %21, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %61 = cherry.tensor_set_slice %arg6[%38, %arg1], %60 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %62 = cherry.reshape %59, %21, %21, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %63 = cherry.tensor_set_slice %arg7[%38, %arg1], %62 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %64 = cherry.tensor_slice %61[%38, %19, %19] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
        %65 = cherry.reshape %64, %27, %22 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
        %66 = cherry.tensor_slice %63[%38, %19, %19] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
        %67 = cherry.reshape %66, %27, %22 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
        %68 = cherry.create_tensor dense<0.000000e+00> : tensor<1x12x64xf32> -> !cherry.cherry_tensor<[1x12x64xf32]>
        %69 = scf.for %arg8 = %c0 to %c12 step %c1 iter_args(%arg9 = %68) -> (!cherry.cherry_tensor<[1x12x64xf32]>) {
          %90 = arith.index_cast %arg8 : index to i64
          %91 = arith.muli %90, %25 : i64
          %92 = cherry.tensor_slice %53[%19, %91] sizes [1, 64] : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x64xf32]>
          %93 = cherry.tensor_slice %65[%19, %91] sizes [1024, 64] : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
          %94 = cherry.transpose %93 perm [1, 0] : (!cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[64x1024xf32]>
          %95 = cherry.masked_matmul %92, %94, %18 : (!cherry.cherry_tensor<[1x64xf32]>, !cherry.cherry_tensor<[64x1024xf32]>, !cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
          %96 = cherry.tensor_mul_scalar %95, %26 : (!cherry.cherry_tensor<[1x1024xf32]>, f32) -> !cherry.cherry_tensor<[1x1024xf32]>
          %97 = cherry.softmax %96 axis 1 : (!cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
          %98 = cherry.tensor_slice %67[%19, %91] sizes [1024, 64] : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
          %99 = cherry.matmul %97, %98 : (!cherry.cherry_tensor<[1x1024xf32]>, !cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[1x64xf32]>
          %100 = cherry.reshape %99, %21, %21, %25 : (!cherry.cherry_tensor<[1x64xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x64xf32]>
          %101 = cherry.tensor_set_slice %arg9[%19, %90], %100 : (!cherry.cherry_tensor<[1x12x64xf32]>, !cherry.cherry_tensor<[1x1x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
          scf.yield %101 : !cherry.cherry_tensor<[1x12x64xf32]>
        }
        %70 = cherry.reshape %69, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %71 = cherry.tensor_slice %5[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %72 = cherry.reshape %71, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %73 = cherry.matmul %70, %72 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %74 = cherry.tensor_add %arg5, %73 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %75 = cherry.tensor_slice %6[%38, %19] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %76 = cherry.reshape %75, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %77 = cherry.rmsnorm %74 scale %76 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %78 = cherry.tensor_slice %7[%38, %19, %19] sizes [1, 768, 2048] : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %79 = cherry.reshape %78, %22, %23 : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
        %80 = cherry.tensor_slice %9[%38, %19, %19] sizes [1, 768, 2048] : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %81 = cherry.reshape %80, %22, %23 : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
        %82 = cherry.matmul %77, %79 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %83 = cherry.matmul %77, %81 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %84 = cherry.tensor_silu %82 : (!cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %85 = cherry.tensor_mul %84, %83 : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %86 = cherry.tensor_slice %8[%38, %19, %19] sizes [1, 2048, 768] : (!cherry.cherry_tensor<[12x2048x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x2048x768xf32]>
        %87 = cherry.reshape %86, %23, %22 : (!cherry.cherry_tensor<[1x2048x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[2048x768xf32]>
        %88 = cherry.matmul %85, %87 : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[2048x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %89 = cherry.tensor_add %74, %88 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        scf.yield %89, %61, %63 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
      }
      %30 = cherry.rmsnorm %29#0 scale %10 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
      %31 = cherry.reshape %11, %22, %28 : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
      %32 = cherry.matmul %30, %31 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
      %33 = cherry.argmax %32 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      cherry.print %32 : !cherry.cherry_tensor<[1x32000xf32]>
      %34 = cherry.constant(0 : i64) : i64
      %35 = cherry.tensor_get %33[%34] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %36 = cherry.constant(1 : i64) : i64
      %37 = arith.addi %arg1, %36 : i64
      scf.yield %35, %37, %29#1, %29#2 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
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
    %5 = cherry.create_tensor dense<7.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %6 = cherry.create_tensor dense<8.000000e+00> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %7 = cherry.create_tensor dense<9.000000e+00> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %8 = cherry.create_tensor dense<1.000000e+01> : tensor<12x2048x768xf32> -> !cherry.cherry_tensor<[12x2048x768xf32]>
    %9 = cherry.create_tensor dense<1.100000e+01> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %10 = cherry.create_tensor dense<1.200000e+01> : tensor<768xf32> -> !cherry.cherry_tensor<[768xf32]>
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
      %18 = cherry.generate_mask %arg1, [1, 1024] : !cherry.cherry_tensor<[1x1024xf32]>
      %19 = cherry.constant(0 : i64) : i64
      %20 = cherry.tensor_slice %0[%arg0, %19] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %21 = cherry.constant(1 : i64) : i64
      %22 = cherry.constant(768 : i64) : i64
      %23 = cherry.constant(2048 : i64) : i64
      %24 = cherry.constant(12 : i64) : i64
      %25 = cherry.constant(64 : i64) : i64
      %26 = cherry.constant(1.250000e-01 : f32) : f32
      %27 = cherry.constant(1024 : i64) : i64
      %28 = cherry.constant(32000 : i64) : i64
      %29:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %20, %arg6 = %arg2, %arg7 = %arg3) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
        %38 = arith.index_cast %arg4 : index to i64
        %39 = cherry.tensor_slice %1[%38, %19] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %40 = cherry.reshape %39, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %41 = cherry.rmsnorm %arg5 scale %40 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %42 = cherry.tensor_slice %2[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %43 = cherry.reshape %42, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %44 = cherry.tensor_slice %3[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %45 = cherry.reshape %44, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %46 = cherry.tensor_slice %4[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %47 = cherry.reshape %46, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %48 = cherry.matmul %41, %43 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %49 = cherry.matmul %41, %45 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %50 = cherry.matmul %41, %47 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %51 = cherry.reshape %48, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %52 = cherry.rope %51, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %53 = cherry.reshape %52, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %54 = cherry.reshape %49, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %55 = cherry.rope %54, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %56 = cherry.reshape %55, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %57 = cherry.reshape %50, %21, %24, %25 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %58 = cherry.rope %57, %arg1 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %59 = cherry.reshape %58, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %60 = cherry.reshape %56, %21, %21, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %61 = cherry.tensor_set_slice %arg6[%38, %arg1], %60 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %62 = cherry.reshape %59, %21, %21, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %63 = cherry.tensor_set_slice %arg7[%38, %arg1], %62 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %64 = cherry.tensor_slice %61[%38, %19, %19] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
        %65 = cherry.reshape %64, %27, %22 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
        %66 = cherry.tensor_slice %63[%38, %19, %19] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
        %67 = cherry.reshape %66, %27, %22 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
        %68 = cherry.create_tensor dense<0.000000e+00> : tensor<1x12x64xf32> -> !cherry.cherry_tensor<[1x12x64xf32]>
        %69 = scf.for %arg8 = %c0 to %c12 step %c1 iter_args(%arg9 = %68) -> (!cherry.cherry_tensor<[1x12x64xf32]>) {
          %90 = arith.index_cast %arg8 : index to i64
          %91 = arith.muli %90, %25 : i64
          %92 = cherry.tensor_slice %53[%19, %91] sizes [1, 64] : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x64xf32]>
          %93 = cherry.tensor_slice %65[%19, %91] sizes [1024, 64] : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
          %94 = cherry.transpose %93 perm [1, 0] : (!cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[64x1024xf32]>
          %95 = cherry.masked_matmul %92, %94, %18 : (!cherry.cherry_tensor<[1x64xf32]>, !cherry.cherry_tensor<[64x1024xf32]>, !cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
          %96 = cherry.tensor_mul_scalar %95, %26 : (!cherry.cherry_tensor<[1x1024xf32]>, f32) -> !cherry.cherry_tensor<[1x1024xf32]>
          %97 = cherry.softmax %96 axis 1 : (!cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
          %98 = cherry.tensor_slice %67[%19, %91] sizes [1024, 64] : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
          %99 = cherry.matmul %97, %98 : (!cherry.cherry_tensor<[1x1024xf32]>, !cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[1x64xf32]>
          %100 = cherry.reshape %99, %21, %21, %25 : (!cherry.cherry_tensor<[1x64xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x64xf32]>
          %101 = cherry.tensor_set_slice %arg9[%19, %90], %100 : (!cherry.cherry_tensor<[1x12x64xf32]>, !cherry.cherry_tensor<[1x1x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
          scf.yield %101 : !cherry.cherry_tensor<[1x12x64xf32]>
        }
        %70 = cherry.reshape %69, %21, %22 : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %71 = cherry.tensor_slice %5[%38, %19, %19] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %72 = cherry.reshape %71, %22, %22 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %73 = cherry.matmul %70, %72 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %74 = cherry.tensor_add %arg5, %73 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %75 = cherry.tensor_slice %6[%38, %19] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %76 = cherry.reshape %75, %22 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %77 = cherry.rmsnorm %74 scale %76 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %78 = cherry.tensor_slice %7[%38, %19, %19] sizes [1, 768, 2048] : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %79 = cherry.reshape %78, %22, %23 : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
        %80 = cherry.tensor_slice %9[%38, %19, %19] sizes [1, 768, 2048] : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %81 = cherry.reshape %80, %22, %23 : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
        %82 = cherry.matmul %77, %79 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %83 = cherry.matmul %77, %81 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %84 = cherry.tensor_silu %82 : (!cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %85 = cherry.tensor_mul %84, %83 : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %86 = cherry.tensor_slice %8[%38, %19, %19] sizes [1, 2048, 768] : (!cherry.cherry_tensor<[12x2048x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x2048x768xf32]>
        %87 = cherry.reshape %86, %23, %22 : (!cherry.cherry_tensor<[1x2048x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[2048x768xf32]>
        %88 = cherry.matmul %85, %87 : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[2048x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %89 = cherry.tensor_add %74, %88 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        scf.yield %89, %61, %63 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
      }
      %30 = cherry.rmsnorm %29#0 scale %10 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
      %31 = cherry.reshape %11, %22, %28 : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
      %32 = cherry.matmul %30, %31 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
      %33 = cherry.argmax %32 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      cherry.print %32 : !cherry.cherry_tensor<[1x32000xf32]>
      %34 = cherry.constant(0 : i64) : i64
      %35 = cherry.tensor_get %33[%34] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %36 = cherry.constant(1 : i64) : i64
      %37 = arith.addi %arg1, %36 : i64
      scf.yield %35, %37, %29#1, %29#2 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
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
#map7 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d2)>
module {
  func.func private @printMemrefF32(tensor<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %cst = arith.constant dense<2.000000e+00> : tensor<32000x768xf32>
    %cst_0 = arith.constant dense<3.000000e+00> : tensor<12x768xf32>
    %cst_1 = arith.constant dense<4.000000e+00> : tensor<12x768x768xf32>
    %cst_2 = arith.constant dense<5.000000e+00> : tensor<12x768x768xf32>
    %cst_3 = arith.constant dense<6.000000e+00> : tensor<12x768x768xf32>
    %cst_4 = arith.constant dense<7.000000e+00> : tensor<12x768x768xf32>
    %cst_5 = arith.constant dense<8.000000e+00> : tensor<12x768xf32>
    %cst_6 = arith.constant dense<9.000000e+00> : tensor<12x768x2048xf32>
    %cst_7 = arith.constant dense<1.000000e+01> : tensor<12x2048x768xf32>
    %cst_8 = arith.constant dense<1.100000e+01> : tensor<12x768x2048xf32>
    %cst_9 = arith.constant dense<1.200000e+01> : tensor<768xf32>
    %cst_10 = arith.constant dense<1.300000e+01> : tensor<32000x768xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<12x1024x768xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<12x1024x768xf32>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10_i64 = arith.constant 10 : i64
    %0:4 = scf.while (%arg0 = %c1_i64, %arg1 = %c0_i64, %arg2 = %cst_11, %arg3 = %cst_12) : (i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) -> (i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) {
      %1 = arith.cmpi slt, %arg1, %c10_i64 : i64
      scf.condition(%1) %arg0, %arg1, %arg2, %arg3 : i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: tensor<12x1024x768xf32>, %arg3: tensor<12x1024x768xf32>):
      %1 = tensor.empty() : tensor<1x1024xf32>
      %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%1 : tensor<1x1024xf32>) {
      ^bb0(%out: f32):
        %25 = linalg.index 1 : index
        %26 = arith.index_cast %25 : index to i64
        %27 = arith.cmpi sle, %26, %arg1 : i64
        %cst_26 = arith.constant 1.000000e+00 : f32
        %cst_27 = arith.constant 0.000000e+00 : f32
        %28 = arith.select %27, %cst_26, %cst_27 : f32
        linalg.yield %28 : f32
      } -> tensor<1x1024xf32>
      %c0_i64_13 = arith.constant 0 : i64
      %3 = arith.index_cast %arg0 : i64 to index
      %4 = arith.index_cast %c0_i64_13 : i64 to index
      %extracted_slice = tensor.extract_slice %cst[%3, %4] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<1x768xf32>
      %c1_i64_14 = arith.constant 1 : i64
      %c768_i64 = arith.constant 768 : i64
      %c2048_i64 = arith.constant 2048 : i64
      %c12_i64 = arith.constant 12 : i64
      %c64_i64 = arith.constant 64 : i64
      %cst_15 = arith.constant 1.250000e-01 : f32
      %c1024_i64 = arith.constant 1024 : i64
      %c32000_i64 = arith.constant 32000 : i64
      %5:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %extracted_slice, %arg6 = %arg2, %arg7 = %arg3) -> (tensor<1x768xf32>, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) {
        %25 = arith.index_cast %arg4 : index to i64
        %26 = arith.index_cast %25 : i64 to index
        %27 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_26 = tensor.extract_slice %cst_0[%26, %27] [1, 768] [1, 1] : tensor<12x768xf32> to tensor<1x768xf32>
        %c768_i64_27 = arith.constant 768 : i64
        %from_elements_28 = tensor.from_elements %c768_i64_27 : tensor<1xi64>
        %reshape_29 = tensor.reshape %extracted_slice_26(%from_elements_28) : (tensor<1x768xf32>, tensor<1xi64>) -> tensor<768xf32>
        %28 = tensor.empty() : tensor<1xf32>
        %cst_30 = arith.constant 0.000000e+00 : f32
        %29 = linalg.fill ins(%cst_30 : f32) outs(%28 : tensor<1xf32>) -> tensor<1xf32>
        %30 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg5 : tensor<1x768xf32>) outs(%29 : tensor<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %130 = arith.mulf %in, %in : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1xf32>
        %c1_31 = arith.constant 1 : index
        %dim_32 = tensor.dim %arg5, %c1_31 : tensor<1x768xf32>
        %31 = arith.index_cast %dim_32 : index to i64
        %32 = arith.uitofp %31 : i64 to f32
        %cst_33 = arith.constant 9.99999974E-6 : f32
        %33 = tensor.empty() : tensor<1xf32>
        %34 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%30 : tensor<1xf32>) outs(%33 : tensor<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %130 = arith.divf %in, %32 : f32
          %131 = arith.addf %130, %cst_33 : f32
          %132 = math.rsqrt %131 : f32
          linalg.yield %132 : f32
        } -> tensor<1xf32>
        %35 = tensor.empty() : tensor<1x768xf32>
        %36 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg5, %34, %reshape_29 : tensor<1x768xf32>, tensor<1xf32>, tensor<768xf32>) outs(%35 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_169: f32, %in_170: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.mulf %130, %in_170 : f32
          linalg.yield %131 : f32
        } -> tensor<1x768xf32>
        %37 = arith.index_cast %25 : i64 to index
        %38 = arith.index_cast %c0_i64_13 : i64 to index
        %39 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_34 = tensor.extract_slice %cst_1[%37, %38, %39] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xf32> to tensor<1x768x768xf32>
        %c768_i64_35 = arith.constant 768 : i64
        %c768_i64_36 = arith.constant 768 : i64
        %from_elements_37 = tensor.from_elements %c768_i64_35, %c768_i64_36 : tensor<2xi64>
        %reshape_38 = tensor.reshape %extracted_slice_34(%from_elements_37) : (tensor<1x768x768xf32>, tensor<2xi64>) -> tensor<768x768xf32>
        %40 = arith.index_cast %25 : i64 to index
        %41 = arith.index_cast %c0_i64_13 : i64 to index
        %42 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_39 = tensor.extract_slice %cst_2[%40, %41, %42] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xf32> to tensor<1x768x768xf32>
        %c768_i64_40 = arith.constant 768 : i64
        %c768_i64_41 = arith.constant 768 : i64
        %from_elements_42 = tensor.from_elements %c768_i64_40, %c768_i64_41 : tensor<2xi64>
        %reshape_43 = tensor.reshape %extracted_slice_39(%from_elements_42) : (tensor<1x768x768xf32>, tensor<2xi64>) -> tensor<768x768xf32>
        %43 = arith.index_cast %25 : i64 to index
        %44 = arith.index_cast %c0_i64_13 : i64 to index
        %45 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_44 = tensor.extract_slice %cst_3[%43, %44, %45] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xf32> to tensor<1x768x768xf32>
        %c768_i64_45 = arith.constant 768 : i64
        %c768_i64_46 = arith.constant 768 : i64
        %from_elements_47 = tensor.from_elements %c768_i64_45, %c768_i64_46 : tensor<2xi64>
        %reshape_48 = tensor.reshape %extracted_slice_44(%from_elements_47) : (tensor<1x768x768xf32>, tensor<2xi64>) -> tensor<768x768xf32>
        %46 = tensor.empty() : tensor<1x768xf32>
        %cst_49 = arith.constant 0.000000e+00 : f32
        %47 = linalg.fill ins(%cst_49 : f32) outs(%46 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %48 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%36, %reshape_38 : tensor<1x768xf32>, tensor<768x768xf32>) outs(%47 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_169: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1x768xf32>
        %49 = tensor.empty() : tensor<1x768xf32>
        %cst_50 = arith.constant 0.000000e+00 : f32
        %50 = linalg.fill ins(%cst_50 : f32) outs(%49 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %51 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%36, %reshape_43 : tensor<1x768xf32>, tensor<768x768xf32>) outs(%50 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_169: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1x768xf32>
        %52 = tensor.empty() : tensor<1x768xf32>
        %cst_51 = arith.constant 0.000000e+00 : f32
        %53 = linalg.fill ins(%cst_51 : f32) outs(%52 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %54 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%36, %reshape_48 : tensor<1x768xf32>, tensor<768x768xf32>) outs(%53 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_169: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1x768xf32>
        %c1_i64_52 = arith.constant 1 : i64
        %c12_i64_53 = arith.constant 12 : i64
        %c64_i64_54 = arith.constant 64 : i64
        %from_elements_55 = tensor.from_elements %c1_i64_52, %c12_i64_53, %c64_i64_54 : tensor<3xi64>
        %reshape_56 = tensor.reshape %48(%from_elements_55) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x12x64xf32>
        %55 = tensor.empty() : tensor<32xf32>
        %56 = tensor.empty() : tensor<32xf32>
        %57 = arith.uitofp %arg1 : i64 to f32
        %58:2 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} outs(%55, %56 : tensor<32xf32>, tensor<32xf32>) {
        ^bb0(%out: f32, %out_169: f32):
          %130 = linalg.index 0 : index
          %131 = arith.index_cast %130 : index to i64
          %132 = arith.uitofp %131 : i64 to f32
          %cst_170 = arith.constant 1.000000e+04 : f32
          %cst_171 = arith.constant 6.400000e+01 : f32
          %cst_172 = arith.constant -2.000000e+00 : f32
          %133 = arith.mulf %cst_172, %132 : f32
          %134 = arith.divf %133, %cst_171 : f32
          %135 = math.powf %cst_170, %134 : f32
          %136 = arith.mulf %57, %135 : f32
          %137 = math.cos %136 : f32
          %138 = math.sin %136 : f32
          linalg.yield %137, %138 : f32, f32
        } -> (tensor<32xf32>, tensor<32xf32>)
        %expanded = tensor.expand_shape %reshape_56 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : tensor<1x12x64xf32> into tensor<1x12x32x2xf32>
        %extracted_slice_57 = tensor.extract_slice %expanded[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %collapsed = tensor.collapse_shape %extracted_slice_57 [[0], [1], [2, 3]] : tensor<1x12x32x1xf32> into tensor<1x12x32xf32>
        %extracted_slice_58 = tensor.extract_slice %expanded[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %collapsed_59 = tensor.collapse_shape %extracted_slice_58 [[0], [1], [2, 3]] : tensor<1x12x32x1xf32> into tensor<1x12x32xf32>
        %59 = tensor.empty() : tensor<1x12x32xf32>
        %60:2 = linalg.generic {indexing_maps = [#map7, #map7, #map8, #map8, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed, %collapsed_59, %58#0, %58#1 : tensor<1x12x32xf32>, tensor<1x12x32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%59, %59 : tensor<1x12x32xf32>, tensor<1x12x32xf32>) {
        ^bb0(%in: f32, %in_169: f32, %in_170: f32, %in_171: f32, %out: f32, %out_172: f32):
          %130 = arith.mulf %in, %in_170 : f32
          %131 = arith.mulf %in_169, %in_171 : f32
          %132 = arith.subf %130, %131 : f32
          %133 = arith.mulf %in_169, %in_170 : f32
          %134 = arith.mulf %in, %in_171 : f32
          %135 = arith.addf %133, %134 : f32
          linalg.yield %132, %135 : f32, f32
        } -> (tensor<1x12x32xf32>, tensor<1x12x32xf32>)
        %expanded_60 = tensor.expand_shape %60#0 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : tensor<1x12x32xf32> into tensor<1x12x32x1xf32>
        %expanded_61 = tensor.expand_shape %60#1 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : tensor<1x12x32xf32> into tensor<1x12x32x1xf32>
        %61 = tensor.empty() : tensor<1x12x32x2xf32>
        %inserted_slice = tensor.insert_slice %expanded_60 into %61[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %inserted_slice_62 = tensor.insert_slice %expanded_61 into %inserted_slice[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %collapsed_63 = tensor.collapse_shape %inserted_slice_62 [[0], [1], [2, 3]] : tensor<1x12x32x2xf32> into tensor<1x12x64xf32>
        %c1_i64_64 = arith.constant 1 : i64
        %c768_i64_65 = arith.constant 768 : i64
        %from_elements_66 = tensor.from_elements %c1_i64_64, %c768_i64_65 : tensor<2xi64>
        %reshape_67 = tensor.reshape %collapsed_63(%from_elements_66) : (tensor<1x12x64xf32>, tensor<2xi64>) -> tensor<1x768xf32>
        %c1_i64_68 = arith.constant 1 : i64
        %c12_i64_69 = arith.constant 12 : i64
        %c64_i64_70 = arith.constant 64 : i64
        %from_elements_71 = tensor.from_elements %c1_i64_68, %c12_i64_69, %c64_i64_70 : tensor<3xi64>
        %reshape_72 = tensor.reshape %51(%from_elements_71) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x12x64xf32>
        %62 = tensor.empty() : tensor<32xf32>
        %63 = tensor.empty() : tensor<32xf32>
        %64 = arith.uitofp %arg1 : i64 to f32
        %65:2 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} outs(%62, %63 : tensor<32xf32>, tensor<32xf32>) {
        ^bb0(%out: f32, %out_169: f32):
          %130 = linalg.index 0 : index
          %131 = arith.index_cast %130 : index to i64
          %132 = arith.uitofp %131 : i64 to f32
          %cst_170 = arith.constant 1.000000e+04 : f32
          %cst_171 = arith.constant 6.400000e+01 : f32
          %cst_172 = arith.constant -2.000000e+00 : f32
          %133 = arith.mulf %cst_172, %132 : f32
          %134 = arith.divf %133, %cst_171 : f32
          %135 = math.powf %cst_170, %134 : f32
          %136 = arith.mulf %64, %135 : f32
          %137 = math.cos %136 : f32
          %138 = math.sin %136 : f32
          linalg.yield %137, %138 : f32, f32
        } -> (tensor<32xf32>, tensor<32xf32>)
        %expanded_73 = tensor.expand_shape %reshape_72 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : tensor<1x12x64xf32> into tensor<1x12x32x2xf32>
        %extracted_slice_74 = tensor.extract_slice %expanded_73[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %collapsed_75 = tensor.collapse_shape %extracted_slice_74 [[0], [1], [2, 3]] : tensor<1x12x32x1xf32> into tensor<1x12x32xf32>
        %extracted_slice_76 = tensor.extract_slice %expanded_73[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %collapsed_77 = tensor.collapse_shape %extracted_slice_76 [[0], [1], [2, 3]] : tensor<1x12x32x1xf32> into tensor<1x12x32xf32>
        %66 = tensor.empty() : tensor<1x12x32xf32>
        %67:2 = linalg.generic {indexing_maps = [#map7, #map7, #map8, #map8, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_75, %collapsed_77, %65#0, %65#1 : tensor<1x12x32xf32>, tensor<1x12x32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%66, %66 : tensor<1x12x32xf32>, tensor<1x12x32xf32>) {
        ^bb0(%in: f32, %in_169: f32, %in_170: f32, %in_171: f32, %out: f32, %out_172: f32):
          %130 = arith.mulf %in, %in_170 : f32
          %131 = arith.mulf %in_169, %in_171 : f32
          %132 = arith.subf %130, %131 : f32
          %133 = arith.mulf %in_169, %in_170 : f32
          %134 = arith.mulf %in, %in_171 : f32
          %135 = arith.addf %133, %134 : f32
          linalg.yield %132, %135 : f32, f32
        } -> (tensor<1x12x32xf32>, tensor<1x12x32xf32>)
        %expanded_78 = tensor.expand_shape %67#0 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : tensor<1x12x32xf32> into tensor<1x12x32x1xf32>
        %expanded_79 = tensor.expand_shape %67#1 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : tensor<1x12x32xf32> into tensor<1x12x32x1xf32>
        %68 = tensor.empty() : tensor<1x12x32x2xf32>
        %inserted_slice_80 = tensor.insert_slice %expanded_78 into %68[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %inserted_slice_81 = tensor.insert_slice %expanded_79 into %inserted_slice_80[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %collapsed_82 = tensor.collapse_shape %inserted_slice_81 [[0], [1], [2, 3]] : tensor<1x12x32x2xf32> into tensor<1x12x64xf32>
        %c1_i64_83 = arith.constant 1 : i64
        %c768_i64_84 = arith.constant 768 : i64
        %from_elements_85 = tensor.from_elements %c1_i64_83, %c768_i64_84 : tensor<2xi64>
        %reshape_86 = tensor.reshape %collapsed_82(%from_elements_85) : (tensor<1x12x64xf32>, tensor<2xi64>) -> tensor<1x768xf32>
        %c1_i64_87 = arith.constant 1 : i64
        %c12_i64_88 = arith.constant 12 : i64
        %c64_i64_89 = arith.constant 64 : i64
        %from_elements_90 = tensor.from_elements %c1_i64_87, %c12_i64_88, %c64_i64_89 : tensor<3xi64>
        %reshape_91 = tensor.reshape %54(%from_elements_90) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x12x64xf32>
        %69 = tensor.empty() : tensor<32xf32>
        %70 = tensor.empty() : tensor<32xf32>
        %71 = arith.uitofp %arg1 : i64 to f32
        %72:2 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} outs(%69, %70 : tensor<32xf32>, tensor<32xf32>) {
        ^bb0(%out: f32, %out_169: f32):
          %130 = linalg.index 0 : index
          %131 = arith.index_cast %130 : index to i64
          %132 = arith.uitofp %131 : i64 to f32
          %cst_170 = arith.constant 1.000000e+04 : f32
          %cst_171 = arith.constant 6.400000e+01 : f32
          %cst_172 = arith.constant -2.000000e+00 : f32
          %133 = arith.mulf %cst_172, %132 : f32
          %134 = arith.divf %133, %cst_171 : f32
          %135 = math.powf %cst_170, %134 : f32
          %136 = arith.mulf %71, %135 : f32
          %137 = math.cos %136 : f32
          %138 = math.sin %136 : f32
          linalg.yield %137, %138 : f32, f32
        } -> (tensor<32xf32>, tensor<32xf32>)
        %expanded_92 = tensor.expand_shape %reshape_91 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : tensor<1x12x64xf32> into tensor<1x12x32x2xf32>
        %extracted_slice_93 = tensor.extract_slice %expanded_92[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %collapsed_94 = tensor.collapse_shape %extracted_slice_93 [[0], [1], [2, 3]] : tensor<1x12x32x1xf32> into tensor<1x12x32xf32>
        %extracted_slice_95 = tensor.extract_slice %expanded_92[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %collapsed_96 = tensor.collapse_shape %extracted_slice_95 [[0], [1], [2, 3]] : tensor<1x12x32x1xf32> into tensor<1x12x32xf32>
        %73 = tensor.empty() : tensor<1x12x32xf32>
        %74:2 = linalg.generic {indexing_maps = [#map7, #map7, #map8, #map8, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_94, %collapsed_96, %72#0, %72#1 : tensor<1x12x32xf32>, tensor<1x12x32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%73, %73 : tensor<1x12x32xf32>, tensor<1x12x32xf32>) {
        ^bb0(%in: f32, %in_169: f32, %in_170: f32, %in_171: f32, %out: f32, %out_172: f32):
          %130 = arith.mulf %in, %in_170 : f32
          %131 = arith.mulf %in_169, %in_171 : f32
          %132 = arith.subf %130, %131 : f32
          %133 = arith.mulf %in_169, %in_170 : f32
          %134 = arith.mulf %in, %in_171 : f32
          %135 = arith.addf %133, %134 : f32
          linalg.yield %132, %135 : f32, f32
        } -> (tensor<1x12x32xf32>, tensor<1x12x32xf32>)
        %expanded_97 = tensor.expand_shape %74#0 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : tensor<1x12x32xf32> into tensor<1x12x32x1xf32>
        %expanded_98 = tensor.expand_shape %74#1 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : tensor<1x12x32xf32> into tensor<1x12x32x1xf32>
        %75 = tensor.empty() : tensor<1x12x32x2xf32>
        %inserted_slice_99 = tensor.insert_slice %expanded_97 into %75[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %inserted_slice_100 = tensor.insert_slice %expanded_98 into %inserted_slice_99[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %collapsed_101 = tensor.collapse_shape %inserted_slice_100 [[0], [1], [2, 3]] : tensor<1x12x32x2xf32> into tensor<1x12x64xf32>
        %c1_i64_102 = arith.constant 1 : i64
        %c768_i64_103 = arith.constant 768 : i64
        %from_elements_104 = tensor.from_elements %c1_i64_102, %c768_i64_103 : tensor<2xi64>
        %reshape_105 = tensor.reshape %collapsed_101(%from_elements_104) : (tensor<1x12x64xf32>, tensor<2xi64>) -> tensor<1x768xf32>
        %c1_i64_106 = arith.constant 1 : i64
        %c1_i64_107 = arith.constant 1 : i64
        %c768_i64_108 = arith.constant 768 : i64
        %from_elements_109 = tensor.from_elements %c1_i64_106, %c1_i64_107, %c768_i64_108 : tensor<3xi64>
        %reshape_110 = tensor.reshape %reshape_86(%from_elements_109) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x1x768xf32>
        %c0_111 = arith.constant 0 : index
        %c1_112 = arith.constant 1 : index
        %76 = arith.index_cast %25 : i64 to index
        %77 = arith.index_cast %arg1 : i64 to index
        %inserted_slice_113 = tensor.insert_slice %reshape_110 into %arg6[%76, %77, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
        %c1_i64_114 = arith.constant 1 : i64
        %c1_i64_115 = arith.constant 1 : i64
        %c768_i64_116 = arith.constant 768 : i64
        %from_elements_117 = tensor.from_elements %c1_i64_114, %c1_i64_115, %c768_i64_116 : tensor<3xi64>
        %reshape_118 = tensor.reshape %reshape_105(%from_elements_117) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x1x768xf32>
        %c0_119 = arith.constant 0 : index
        %c1_120 = arith.constant 1 : index
        %78 = arith.index_cast %25 : i64 to index
        %79 = arith.index_cast %arg1 : i64 to index
        %inserted_slice_121 = tensor.insert_slice %reshape_118 into %arg7[%78, %79, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
        %80 = arith.index_cast %25 : i64 to index
        %81 = arith.index_cast %c0_i64_13 : i64 to index
        %82 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_122 = tensor.extract_slice %inserted_slice_113[%80, %81, %82] [1, 1024, 768] [1, 1, 1] : tensor<12x1024x768xf32> to tensor<1x1024x768xf32>
        %c1024_i64_123 = arith.constant 1024 : i64
        %c768_i64_124 = arith.constant 768 : i64
        %from_elements_125 = tensor.from_elements %c1024_i64_123, %c768_i64_124 : tensor<2xi64>
        %reshape_126 = tensor.reshape %extracted_slice_122(%from_elements_125) : (tensor<1x1024x768xf32>, tensor<2xi64>) -> tensor<1024x768xf32>
        %83 = arith.index_cast %25 : i64 to index
        %84 = arith.index_cast %c0_i64_13 : i64 to index
        %85 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_127 = tensor.extract_slice %inserted_slice_121[%83, %84, %85] [1, 1024, 768] [1, 1, 1] : tensor<12x1024x768xf32> to tensor<1x1024x768xf32>
        %c1024_i64_128 = arith.constant 1024 : i64
        %c768_i64_129 = arith.constant 768 : i64
        %from_elements_130 = tensor.from_elements %c1024_i64_128, %c768_i64_129 : tensor<2xi64>
        %reshape_131 = tensor.reshape %extracted_slice_127(%from_elements_130) : (tensor<1x1024x768xf32>, tensor<2xi64>) -> tensor<1024x768xf32>
        %cst_132 = arith.constant dense<0.000000e+00> : tensor<1x12x64xf32>
        %86 = scf.for %arg8 = %c0 to %c12 step %c1 iter_args(%arg9 = %cst_132) -> (tensor<1x12x64xf32>) {
          %130 = arith.index_cast %arg8 : index to i64
          %131 = arith.muli %130, %c64_i64 : i64
          %132 = arith.index_cast %c0_i64_13 : i64 to index
          %133 = arith.index_cast %131 : i64 to index
          %extracted_slice_169 = tensor.extract_slice %reshape_67[%132, %133] [1, 64] [1, 1] : tensor<1x768xf32> to tensor<1x64xf32>
          %134 = arith.index_cast %c0_i64_13 : i64 to index
          %135 = arith.index_cast %131 : i64 to index
          %extracted_slice_170 = tensor.extract_slice %reshape_126[%134, %135] [1024, 64] [1, 1] : tensor<1024x768xf32> to tensor<1024x64xf32>
          %136 = tensor.empty() : tensor<64x1024xf32>
          %transposed = linalg.transpose ins(%extracted_slice_170 : tensor<1024x64xf32>) outs(%136 : tensor<64x1024xf32>) permutation = [1, 0] 
          %137 = tensor.empty() : tensor<1x1024xf32>
          %cst_171 = arith.constant 0.000000e+00 : f32
          %138 = linalg.fill ins(%cst_171 : f32) outs(%137 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
          %139 = linalg.generic {indexing_maps = [#map4, #map5, #map6, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_169, %transposed, %2 : tensor<1x64xf32>, tensor<64x1024xf32>, tensor<1x1024xf32>) outs(%138 : tensor<1x1024xf32>) {
          ^bb0(%in: f32, %in_184: f32, %in_185: f32, %out: f32):
            %159 = arith.mulf %in, %in_184 : f32
            %160 = arith.addf %out, %159 : f32
            %cst_186 = arith.constant -1.000000e+09 : f32
            %cst_187 = arith.constant 5.000000e-01 : f32
            %161 = arith.cmpf ugt, %in_185, %cst_187 : f32
            %162 = arith.select %161, %160, %cst_186 : f32
            linalg.yield %162 : f32
          } -> tensor<1x1024xf32>
          %140 = tensor.empty() : tensor<1x1024xf32>
          %141 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%139 : tensor<1x1024xf32>) outs(%140 : tensor<1x1024xf32>) {
          ^bb0(%in: f32, %out: f32):
            %159 = arith.mulf %in, %cst_15 : f32
            linalg.yield %159 : f32
          } -> tensor<1x1024xf32>
          %142 = tensor.empty() : tensor<1xf32>
          %cst_172 = arith.constant 0xFF800000 : f32
          %143 = linalg.fill ins(%cst_172 : f32) outs(%142 : tensor<1xf32>) -> tensor<1xf32>
          %144 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%141 : tensor<1x1024xf32>) outs(%143 : tensor<1xf32>) {
          ^bb0(%in: f32, %out: f32):
            %159 = arith.maxnumf %in, %out : f32
            linalg.yield %159 : f32
          } -> tensor<1xf32>
          %145 = tensor.empty() : tensor<1x1024xf32>
          %146 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%141, %144 : tensor<1x1024xf32>, tensor<1xf32>) outs(%145 : tensor<1x1024xf32>) {
          ^bb0(%in: f32, %in_184: f32, %out: f32):
            %159 = arith.subf %in, %in_184 : f32
            %160 = math.exp %159 : f32
            linalg.yield %160 : f32
          } -> tensor<1x1024xf32>
          %147 = tensor.empty() : tensor<1xf32>
          %cst_173 = arith.constant 0.000000e+00 : f32
          %148 = linalg.fill ins(%cst_173 : f32) outs(%147 : tensor<1xf32>) -> tensor<1xf32>
          %149 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%146 : tensor<1x1024xf32>) outs(%148 : tensor<1xf32>) {
          ^bb0(%in: f32, %out: f32):
            %159 = arith.addf %in, %out : f32
            linalg.yield %159 : f32
          } -> tensor<1xf32>
          %150 = tensor.empty() : tensor<1x1024xf32>
          %151 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%146, %149 : tensor<1x1024xf32>, tensor<1xf32>) outs(%150 : tensor<1x1024xf32>) {
          ^bb0(%in: f32, %in_184: f32, %out: f32):
            %159 = arith.divf %in, %in_184 : f32
            linalg.yield %159 : f32
          } -> tensor<1x1024xf32>
          %152 = arith.index_cast %c0_i64_13 : i64 to index
          %153 = arith.index_cast %131 : i64 to index
          %extracted_slice_174 = tensor.extract_slice %reshape_131[%152, %153] [1024, 64] [1, 1] : tensor<1024x768xf32> to tensor<1024x64xf32>
          %154 = tensor.empty() : tensor<1x64xf32>
          %cst_175 = arith.constant 0.000000e+00 : f32
          %155 = linalg.fill ins(%cst_175 : f32) outs(%154 : tensor<1x64xf32>) -> tensor<1x64xf32>
          %156 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%151, %extracted_slice_174 : tensor<1x1024xf32>, tensor<1024x64xf32>) outs(%155 : tensor<1x64xf32>) {
          ^bb0(%in: f32, %in_184: f32, %out: f32):
            %159 = arith.mulf %in, %in_184 : f32
            %160 = arith.addf %out, %159 : f32
            linalg.yield %160 : f32
          } -> tensor<1x64xf32>
          %c1_i64_176 = arith.constant 1 : i64
          %c1_i64_177 = arith.constant 1 : i64
          %c64_i64_178 = arith.constant 64 : i64
          %from_elements_179 = tensor.from_elements %c1_i64_176, %c1_i64_177, %c64_i64_178 : tensor<3xi64>
          %reshape_180 = tensor.reshape %156(%from_elements_179) : (tensor<1x64xf32>, tensor<3xi64>) -> tensor<1x1x64xf32>
          %c0_181 = arith.constant 0 : index
          %c1_182 = arith.constant 1 : index
          %157 = arith.index_cast %c0_i64_13 : i64 to index
          %158 = arith.index_cast %130 : i64 to index
          %inserted_slice_183 = tensor.insert_slice %reshape_180 into %arg9[%157, %158, 0] [1, 1, 64] [1, 1, 1] : tensor<1x1x64xf32> into tensor<1x12x64xf32>
          scf.yield %inserted_slice_183 : tensor<1x12x64xf32>
        }
        %c1_i64_133 = arith.constant 1 : i64
        %c768_i64_134 = arith.constant 768 : i64
        %from_elements_135 = tensor.from_elements %c1_i64_133, %c768_i64_134 : tensor<2xi64>
        %reshape_136 = tensor.reshape %86(%from_elements_135) : (tensor<1x12x64xf32>, tensor<2xi64>) -> tensor<1x768xf32>
        %87 = arith.index_cast %25 : i64 to index
        %88 = arith.index_cast %c0_i64_13 : i64 to index
        %89 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_137 = tensor.extract_slice %cst_4[%87, %88, %89] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xf32> to tensor<1x768x768xf32>
        %c768_i64_138 = arith.constant 768 : i64
        %c768_i64_139 = arith.constant 768 : i64
        %from_elements_140 = tensor.from_elements %c768_i64_138, %c768_i64_139 : tensor<2xi64>
        %reshape_141 = tensor.reshape %extracted_slice_137(%from_elements_140) : (tensor<1x768x768xf32>, tensor<2xi64>) -> tensor<768x768xf32>
        %90 = tensor.empty() : tensor<1x768xf32>
        %cst_142 = arith.constant 0.000000e+00 : f32
        %91 = linalg.fill ins(%cst_142 : f32) outs(%90 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %92 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%reshape_136, %reshape_141 : tensor<1x768xf32>, tensor<768x768xf32>) outs(%91 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_169: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1x768xf32>
        %93 = tensor.empty() : tensor<1x768xf32>
        %94 = linalg.add ins(%arg5, %92 : tensor<1x768xf32>, tensor<1x768xf32>) outs(%93 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %95 = arith.index_cast %25 : i64 to index
        %96 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_143 = tensor.extract_slice %cst_5[%95, %96] [1, 768] [1, 1] : tensor<12x768xf32> to tensor<1x768xf32>
        %c768_i64_144 = arith.constant 768 : i64
        %from_elements_145 = tensor.from_elements %c768_i64_144 : tensor<1xi64>
        %reshape_146 = tensor.reshape %extracted_slice_143(%from_elements_145) : (tensor<1x768xf32>, tensor<1xi64>) -> tensor<768xf32>
        %97 = tensor.empty() : tensor<1xf32>
        %cst_147 = arith.constant 0.000000e+00 : f32
        %98 = linalg.fill ins(%cst_147 : f32) outs(%97 : tensor<1xf32>) -> tensor<1xf32>
        %99 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%94 : tensor<1x768xf32>) outs(%98 : tensor<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %130 = arith.mulf %in, %in : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1xf32>
        %c1_148 = arith.constant 1 : index
        %dim_149 = tensor.dim %94, %c1_148 : tensor<1x768xf32>
        %100 = arith.index_cast %dim_149 : index to i64
        %101 = arith.uitofp %100 : i64 to f32
        %cst_150 = arith.constant 9.99999974E-6 : f32
        %102 = tensor.empty() : tensor<1xf32>
        %103 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%99 : tensor<1xf32>) outs(%102 : tensor<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %130 = arith.divf %in, %101 : f32
          %131 = arith.addf %130, %cst_150 : f32
          %132 = math.rsqrt %131 : f32
          linalg.yield %132 : f32
        } -> tensor<1xf32>
        %104 = tensor.empty() : tensor<1x768xf32>
        %105 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%94, %103, %reshape_146 : tensor<1x768xf32>, tensor<1xf32>, tensor<768xf32>) outs(%104 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_169: f32, %in_170: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.mulf %130, %in_170 : f32
          linalg.yield %131 : f32
        } -> tensor<1x768xf32>
        %106 = arith.index_cast %25 : i64 to index
        %107 = arith.index_cast %c0_i64_13 : i64 to index
        %108 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_151 = tensor.extract_slice %cst_6[%106, %107, %108] [1, 768, 2048] [1, 1, 1] : tensor<12x768x2048xf32> to tensor<1x768x2048xf32>
        %c768_i64_152 = arith.constant 768 : i64
        %c2048_i64_153 = arith.constant 2048 : i64
        %from_elements_154 = tensor.from_elements %c768_i64_152, %c2048_i64_153 : tensor<2xi64>
        %reshape_155 = tensor.reshape %extracted_slice_151(%from_elements_154) : (tensor<1x768x2048xf32>, tensor<2xi64>) -> tensor<768x2048xf32>
        %109 = arith.index_cast %25 : i64 to index
        %110 = arith.index_cast %c0_i64_13 : i64 to index
        %111 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_156 = tensor.extract_slice %cst_8[%109, %110, %111] [1, 768, 2048] [1, 1, 1] : tensor<12x768x2048xf32> to tensor<1x768x2048xf32>
        %c768_i64_157 = arith.constant 768 : i64
        %c2048_i64_158 = arith.constant 2048 : i64
        %from_elements_159 = tensor.from_elements %c768_i64_157, %c2048_i64_158 : tensor<2xi64>
        %reshape_160 = tensor.reshape %extracted_slice_156(%from_elements_159) : (tensor<1x768x2048xf32>, tensor<2xi64>) -> tensor<768x2048xf32>
        %112 = tensor.empty() : tensor<1x2048xf32>
        %cst_161 = arith.constant 0.000000e+00 : f32
        %113 = linalg.fill ins(%cst_161 : f32) outs(%112 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
        %114 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%105, %reshape_155 : tensor<1x768xf32>, tensor<768x2048xf32>) outs(%113 : tensor<1x2048xf32>) {
        ^bb0(%in: f32, %in_169: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1x2048xf32>
        %115 = tensor.empty() : tensor<1x2048xf32>
        %cst_162 = arith.constant 0.000000e+00 : f32
        %116 = linalg.fill ins(%cst_162 : f32) outs(%115 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
        %117 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%105, %reshape_160 : tensor<1x768xf32>, tensor<768x2048xf32>) outs(%116 : tensor<1x2048xf32>) {
        ^bb0(%in: f32, %in_169: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1x2048xf32>
        %118 = tensor.empty() : tensor<1x2048xf32>
        %119 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%114 : tensor<1x2048xf32>) outs(%118 : tensor<1x2048xf32>) {
        ^bb0(%in: f32, %out: f32):
          %130 = arith.negf %in : f32
          %131 = math.exp %130 : f32
          %132 = arith.addf %in, %131 : f32
          %133 = arith.divf %in, %132 : f32
          linalg.yield %133 : f32
        } -> tensor<1x2048xf32>
        %120 = tensor.empty() : tensor<1x2048xf32>
        %121 = linalg.mul ins(%119, %117 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%120 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
        %122 = arith.index_cast %25 : i64 to index
        %123 = arith.index_cast %c0_i64_13 : i64 to index
        %124 = arith.index_cast %c0_i64_13 : i64 to index
        %extracted_slice_163 = tensor.extract_slice %cst_7[%122, %123, %124] [1, 2048, 768] [1, 1, 1] : tensor<12x2048x768xf32> to tensor<1x2048x768xf32>
        %c2048_i64_164 = arith.constant 2048 : i64
        %c768_i64_165 = arith.constant 768 : i64
        %from_elements_166 = tensor.from_elements %c2048_i64_164, %c768_i64_165 : tensor<2xi64>
        %reshape_167 = tensor.reshape %extracted_slice_163(%from_elements_166) : (tensor<1x2048x768xf32>, tensor<2xi64>) -> tensor<2048x768xf32>
        %125 = tensor.empty() : tensor<1x768xf32>
        %cst_168 = arith.constant 0.000000e+00 : f32
        %126 = linalg.fill ins(%cst_168 : f32) outs(%125 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %127 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121, %reshape_167 : tensor<1x2048xf32>, tensor<2048x768xf32>) outs(%126 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_169: f32, %out: f32):
          %130 = arith.mulf %in, %in_169 : f32
          %131 = arith.addf %out, %130 : f32
          linalg.yield %131 : f32
        } -> tensor<1x768xf32>
        %128 = tensor.empty() : tensor<1x768xf32>
        %129 = linalg.add ins(%94, %127 : tensor<1x768xf32>, tensor<1x768xf32>) outs(%128 : tensor<1x768xf32>) -> tensor<1x768xf32>
        scf.yield %129, %inserted_slice_113, %inserted_slice_121 : tensor<1x768xf32>, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
      }
      %6 = tensor.empty() : tensor<1xf32>
      %cst_16 = arith.constant 0.000000e+00 : f32
      %7 = linalg.fill ins(%cst_16 : f32) outs(%6 : tensor<1xf32>) -> tensor<1xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%5#0 : tensor<1x768xf32>) outs(%7 : tensor<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %25 = arith.mulf %in, %in : f32
        %26 = arith.addf %out, %25 : f32
        linalg.yield %26 : f32
      } -> tensor<1xf32>
      %c1_17 = arith.constant 1 : index
      %dim = tensor.dim %5#0, %c1_17 : tensor<1x768xf32>
      %9 = arith.index_cast %dim : index to i64
      %10 = arith.uitofp %9 : i64 to f32
      %cst_18 = arith.constant 9.99999974E-6 : f32
      %11 = tensor.empty() : tensor<1xf32>
      %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%8 : tensor<1xf32>) outs(%11 : tensor<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %25 = arith.divf %in, %10 : f32
        %26 = arith.addf %25, %cst_18 : f32
        %27 = math.rsqrt %26 : f32
        linalg.yield %27 : f32
      } -> tensor<1xf32>
      %13 = tensor.empty() : tensor<1x768xf32>
      %14 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%5#0, %12, %cst_9 : tensor<1x768xf32>, tensor<1xf32>, tensor<768xf32>) outs(%13 : tensor<1x768xf32>) {
      ^bb0(%in: f32, %in_26: f32, %in_27: f32, %out: f32):
        %25 = arith.mulf %in, %in_26 : f32
        %26 = arith.mulf %25, %in_27 : f32
        linalg.yield %26 : f32
      } -> tensor<1x768xf32>
      %c768_i64_19 = arith.constant 768 : i64
      %c32000_i64_20 = arith.constant 32000 : i64
      %from_elements = tensor.from_elements %c768_i64_19, %c32000_i64_20 : tensor<2xi64>
      %reshape = tensor.reshape %cst_10(%from_elements) : (tensor<32000x768xf32>, tensor<2xi64>) -> tensor<768x32000xf32>
      %15 = tensor.empty() : tensor<1x32000xf32>
      %cst_21 = arith.constant 0.000000e+00 : f32
      %16 = linalg.fill ins(%cst_21 : f32) outs(%15 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
      %17 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14, %reshape : tensor<1x768xf32>, tensor<768x32000xf32>) outs(%16 : tensor<1x32000xf32>) {
      ^bb0(%in: f32, %in_26: f32, %out: f32):
        %25 = arith.mulf %in, %in_26 : f32
        %26 = arith.addf %out, %25 : f32
        linalg.yield %26 : f32
      } -> tensor<1x32000xf32>
      %cst_22 = arith.constant 0xFF800000 : f32
      %18 = tensor.empty() : tensor<1xf32>
      %19 = linalg.fill ins(%cst_22 : f32) outs(%18 : tensor<1xf32>) -> tensor<1xf32>
      %c0_i64_23 = arith.constant 0 : i64
      %20 = tensor.empty() : tensor<1xi64>
      %21 = linalg.fill ins(%c0_i64_23 : i64) outs(%20 : tensor<1xi64>) -> tensor<1xi64>
      %22:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%17 : tensor<1x32000xf32>) outs(%19, %21 : tensor<1xf32>, tensor<1xi64>) {
      ^bb0(%in: f32, %out: f32, %out_26: i64):
        %25 = linalg.index 1 : index
        %26 = arith.index_cast %25 : index to i64
        %27 = arith.cmpf ogt, %in, %out : f32
        %28 = arith.select %27, %in, %out : f32
        %29 = arith.select %27, %26, %out_26 : i64
        linalg.yield %28, %29 : f32, i64
      } -> (tensor<1xf32>, tensor<1xi64>)
      %cast = tensor.cast %17 : tensor<1x32000xf32> to tensor<*xf32>
      func.call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
      %c0_i64_24 = arith.constant 0 : i64
      %23 = arith.index_cast %c0_i64_24 : i64 to index
      %extracted = tensor.extract %22#1[%23] : tensor<1xi64>
      %c1_i64_25 = arith.constant 1 : i64
      %24 = arith.addi %arg1, %c1_i64_25 : i64
      scf.yield %extracted, %24, %5#1, %5#2 : i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
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
#map7 = affine_map<(d0) -> (-d0 + 12, 8)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map10 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d0)>
#map12 = affine_map<(d0, d1, d2) -> (d2, d1)>
module {
  func.func private @printMemrefF32(tensor<*xf32> {bufferization.access = "read"})
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
    %c8_41 = arith.constant 8 : index
    %c8_42 = arith.constant 8 : index
    %c8_43 = arith.constant 8 : index
    %c8_44 = arith.constant 8 : index
    %c8_45 = arith.constant 8 : index
    %c8_46 = arith.constant 8 : index
    %c8_47 = arith.constant 8 : index
    %c8_48 = arith.constant 8 : index
    %c8_49 = arith.constant 8 : index
    %c8_50 = arith.constant 8 : index
    %c8_51 = arith.constant 8 : index
    %c8_52 = arith.constant 8 : index
    %c8_53 = arith.constant 8 : index
    %c8_54 = arith.constant 8 : index
    %c8_55 = arith.constant 8 : index
    %c8_56 = arith.constant 8 : index
    %c8_57 = arith.constant 8 : index
    %c8_58 = arith.constant 8 : index
    %c8_59 = arith.constant 8 : index
    %c8_60 = arith.constant 8 : index
    %c8_61 = arith.constant 8 : index
    %c8_62 = arith.constant 8 : index
    %c8_63 = arith.constant 8 : index
    %c8_64 = arith.constant 8 : index
    %c8_65 = arith.constant 8 : index
    %c8_66 = arith.constant 8 : index
    %c8_67 = arith.constant 8 : index
    %c8_68 = arith.constant 8 : index
    %c8_69 = arith.constant 8 : index
    %c8_70 = arith.constant 8 : index
    %c8_71 = arith.constant 8 : index
    %c8_72 = arith.constant 8 : index
    %c8_73 = arith.constant 8 : index
    %c8_74 = arith.constant 8 : index
    %c8_75 = arith.constant 8 : index
    %c8_76 = arith.constant 8 : index
    %c8_77 = arith.constant 8 : index
    %c8_78 = arith.constant 8 : index
    %c8_79 = arith.constant 8 : index
    %c8_80 = arith.constant 8 : index
    %c8_81 = arith.constant 8 : index
    %c8_82 = arith.constant 8 : index
    %c8_83 = arith.constant 8 : index
    %c8_84 = arith.constant 8 : index
    %c8_85 = arith.constant 8 : index
    %c8_86 = arith.constant 8 : index
    %c8_87 = arith.constant 8 : index
    %c8_88 = arith.constant 8 : index
    %c8_89 = arith.constant 8 : index
    %c8_90 = arith.constant 8 : index
    %c8_91 = arith.constant 8 : index
    %c8_92 = arith.constant 8 : index
    %c8_93 = arith.constant 8 : index
    %c8_94 = arith.constant 8 : index
    %c8_95 = arith.constant 8 : index
    %c8_96 = arith.constant 8 : index
    %c8_97 = arith.constant 8 : index
    %c8_98 = arith.constant 8 : index
    %c8_99 = arith.constant 8 : index
    %c8_100 = arith.constant 8 : index
    %c8_101 = arith.constant 8 : index
    %c8_102 = arith.constant 8 : index
    %c8_103 = arith.constant 8 : index
    %c8_104 = arith.constant 8 : index
    %c8_105 = arith.constant 8 : index
    %c8_106 = arith.constant 8 : index
    %c8_107 = arith.constant 8 : index
    %c8_108 = arith.constant 8 : index
    %c8_109 = arith.constant 8 : index
    %c8_110 = arith.constant 8 : index
    %c8_111 = arith.constant 8 : index
    %c8_112 = arith.constant 8 : index
    %c8_113 = arith.constant 8 : index
    %c8_114 = arith.constant 8 : index
    %c8_115 = arith.constant 8 : index
    %c8_116 = arith.constant 8 : index
    %c8_117 = arith.constant 8 : index
    %c8_118 = arith.constant 8 : index
    %c8_119 = arith.constant 8 : index
    %c8_120 = arith.constant 8 : index
    %c8_121 = arith.constant 8 : index
    %c8_122 = arith.constant 8 : index
    %c8_123 = arith.constant 8 : index
    %c8_124 = arith.constant 8 : index
    %c8_125 = arith.constant 8 : index
    %c8_126 = arith.constant 8 : index
    %c8_127 = arith.constant 8 : index
    %c8_128 = arith.constant 8 : index
    %c8_129 = arith.constant 8 : index
    %c8_130 = arith.constant 8 : index
    %c8_131 = arith.constant 8 : index
    %c8_132 = arith.constant 8 : index
    %c8_133 = arith.constant 8 : index
    %cst = arith.constant 1.300000e+01 : f32
    %cst_134 = arith.constant 1.200000e+01 : f32
    %cst_135 = arith.constant 1.000000e+01 : f32
    %cst_136 = arith.constant 1.100000e+01 : f32
    %cst_137 = arith.constant 9.000000e+00 : f32
    %cst_138 = arith.constant 8.000000e+00 : f32
    %cst_139 = arith.constant 7.000000e+00 : f32
    %cst_140 = arith.constant 0xFF800000 : f32
    %cst_141 = arith.constant 6.000000e+00 : f32
    %cst_142 = arith.constant 5.000000e+00 : f32
    %cst_143 = arith.constant 4.000000e+00 : f32
    %cst_144 = arith.constant 3.000000e+00 : f32
    %cst_145 = arith.constant dense<[1, 1, 64]> : tensor<3xi64>
    %cst_146 = arith.constant dense<[1024, 768]> : tensor<2xi64>
    %cst_147 = arith.constant dense<[1, 1, 768]> : tensor<3xi64>
    %cst_148 = arith.constant dense<[1, 768]> : tensor<2xi64>
    %cst_149 = arith.constant dense<[1, 12, 64]> : tensor<3xi64>
    %cst_150 = arith.constant 7.680000e+02 : f32
    %cst_151 = arith.constant dense<2.000000e+00> : tensor<1x768xf32>
    %cst_152 = arith.constant 5.000000e-01 : f32
    %cst_153 = arith.constant -1.000000e+09 : f32
    %cst_154 = arith.constant dense<0.000000e+00> : tensor<1x12x64xf32>
    %cst_155 = arith.constant -2.000000e+00 : f32
    %cst_156 = arith.constant 6.400000e+01 : f32
    %cst_157 = arith.constant 1.000000e+04 : f32
    %cst_158 = arith.constant 9.99999974E-6 : f32
    %cst_159 = arith.constant 1.250000e-01 : f32
    %c64_i64 = arith.constant 64 : i64
    %cst_160 = arith.constant 0.000000e+00 : f32
    %cst_161 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %cst_162 = arith.constant dense<0.000000e+00> : tensor<12x1024x768xf32>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10_i64 = arith.constant 10 : i64
    %0:4 = scf.while (%arg0 = %c1_i64, %arg1 = %c0_i64, %arg2 = %cst_162, %arg3 = %cst_162) : (i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) -> (i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) {
      %1 = arith.cmpi slt, %arg1, %c10_i64 : i64
      scf.condition(%1) %arg0, %arg1, %arg2, %arg3 : i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: tensor<12x1024x768xf32>, %arg3: tensor<12x1024x768xf32>):
      %1:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %cst_151, %arg6 = %arg2, %arg7 = %arg3) -> (tensor<1x768xf32>, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) {
        %14 = arith.index_cast %arg4 : index to i64
        %15 = tensor.empty() : tensor<1xf32>
        %c0_197 = arith.constant 0 : index
        %c1_198 = arith.constant 1 : index
        %c8_199 = arith.constant 8 : index
        %16 = scf.for %arg8 = %c0_197 to %c1_198 step %c8_199 iter_args(%arg9 = %15) -> (tensor<1xf32>) {
          %81 = affine.min #map(%arg8)
          %extracted_slice_410 = tensor.extract_slice %arg9[%arg8] [%81] [1] : tensor<1xf32> to tensor<?xf32>
          %82 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_410 : tensor<?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_160 : f32
          } -> tensor<?xf32>
          %inserted_slice_411 = tensor.insert_slice %82 into %arg9[%arg8] [%81] [1] : tensor<?xf32> into tensor<1xf32>
          scf.yield %inserted_slice_411 : tensor<1xf32>
        }
        %c0_200 = arith.constant 0 : index
        %c1_201 = arith.constant 1 : index
        %c8_202 = arith.constant 8 : index
        %c0_203 = arith.constant 0 : index
        %c768_204 = arith.constant 768 : index
        %c8_205 = arith.constant 8 : index
        %17 = scf.for %arg8 = %c0_200 to %c1_201 step %c8_202 iter_args(%arg9 = %16) -> (tensor<1xf32>) {
          %81 = scf.for %arg10 = %c0_203 to %c768_204 step %c8_205 iter_args(%arg11 = %arg9) -> (tensor<1xf32>) {
            %82 = affine.min #map(%arg8)
            %83 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg5[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_411 = tensor.extract_slice %arg11[%arg8] [%83] [1] : tensor<1xf32> to tensor<?xf32>
            %84 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_410 : tensor<?x8xf32>) outs(%extracted_slice_411 : tensor<?xf32>) {
            ^bb0(%in: f32, %out: f32):
              %85 = arith.mulf %in, %in : f32
              %86 = arith.addf %out, %85 : f32
              linalg.yield %86 : f32
            } -> tensor<?xf32>
            %inserted_slice_412 = tensor.insert_slice %84 into %arg11[%arg8] [%83] [1] : tensor<?xf32> into tensor<1xf32>
            scf.yield %inserted_slice_412 : tensor<1xf32>
          }
          scf.yield %81 : tensor<1xf32>
        }
        %18 = tensor.empty() : tensor<1x768xf32>
        %c0_206 = arith.constant 0 : index
        %c1_207 = arith.constant 1 : index
        %c8_208 = arith.constant 8 : index
        %c0_209 = arith.constant 0 : index
        %c768_210 = arith.constant 768 : index
        %c8_211 = arith.constant 8 : index
        %19 = scf.for %arg8 = %c0_206 to %c1_207 step %c8_208 iter_args(%arg9 = %18) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_209 to %c768_210 step %c8_211 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %83 = affine.min #map(%arg8)
            %84 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg5[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_411 = tensor.extract_slice %17[%arg8] [%83] [1] : tensor<1xf32> to tensor<?xf32>
            %extracted_slice_412 = tensor.extract_slice %arg11[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %85 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_410, %extracted_slice_411 : tensor<?x8xf32>, tensor<?xf32>) outs(%extracted_slice_412 : tensor<?x8xf32>) {
            ^bb0(%in: f32, %in_414: f32, %out: f32):
              %86 = arith.divf %in_414, %cst_150 : f32
              %87 = arith.addf %86, %cst_158 : f32
              %88 = math.rsqrt %87 : f32
              %89 = arith.mulf %in, %88 : f32
              %90 = arith.mulf %89, %cst_144 : f32
              linalg.yield %90 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_413 = tensor.insert_slice %85 into %arg11[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_413 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %20 = tensor.empty() : tensor<1x768xf32>
        %c0_212 = arith.constant 0 : index
        %c1_213 = arith.constant 1 : index
        %c8_214 = arith.constant 8 : index
        %c0_215 = arith.constant 0 : index
        %c768_216 = arith.constant 768 : index
        %c8_217 = arith.constant 8 : index
        %21 = scf.for %arg8 = %c0_212 to %c1_213 step %c8_214 iter_args(%arg9 = %20) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_215 to %c768_216 step %c8_217 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %83 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_410 : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_160 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_411 = tensor.insert_slice %83 into %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_411 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %c0_218 = arith.constant 0 : index
        %c1_219 = arith.constant 1 : index
        %c8_220 = arith.constant 8 : index
        %c0_221 = arith.constant 0 : index
        %c768_222 = arith.constant 768 : index
        %c8_223 = arith.constant 8 : index
        %c0_224 = arith.constant 0 : index
        %c768_225 = arith.constant 768 : index
        %c8_226 = arith.constant 8 : index
        %22 = scf.for %arg8 = %c0_218 to %c1_219 step %c8_220 iter_args(%arg9 = %21) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_221 to %c768_222 step %c8_223 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = scf.for %arg12 = %c0_224 to %c768_225 step %c8_226 iter_args(%arg13 = %arg11) -> (tensor<1x768xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map(%arg8)
              %extracted_slice_410 = tensor.extract_slice %19[%arg8, %arg12] [%83, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_411 = tensor.extract_slice %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %85 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_410 : tensor<?x8xf32>) outs(%extracted_slice_411 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %86 = arith.mulf %in, %cst_143 : f32
                %87 = arith.addf %out, %86 : f32
                linalg.yield %87 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_412 = tensor.insert_slice %85 into %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
              scf.yield %inserted_slice_412 : tensor<1x768xf32>
            }
            scf.yield %82 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %23 = tensor.empty() : tensor<1x768xf32>
        %c0_227 = arith.constant 0 : index
        %c1_228 = arith.constant 1 : index
        %c8_229 = arith.constant 8 : index
        %c0_230 = arith.constant 0 : index
        %c768_231 = arith.constant 768 : index
        %c8_232 = arith.constant 8 : index
        %24 = scf.for %arg8 = %c0_227 to %c1_228 step %c8_229 iter_args(%arg9 = %23) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_230 to %c768_231 step %c8_232 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %83 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_410 : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_160 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_411 = tensor.insert_slice %83 into %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_411 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %c0_233 = arith.constant 0 : index
        %c1_234 = arith.constant 1 : index
        %c8_235 = arith.constant 8 : index
        %c0_236 = arith.constant 0 : index
        %c768_237 = arith.constant 768 : index
        %c8_238 = arith.constant 8 : index
        %c0_239 = arith.constant 0 : index
        %c768_240 = arith.constant 768 : index
        %c8_241 = arith.constant 8 : index
        %25 = scf.for %arg8 = %c0_233 to %c1_234 step %c8_235 iter_args(%arg9 = %24) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_236 to %c768_237 step %c8_238 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = scf.for %arg12 = %c0_239 to %c768_240 step %c8_241 iter_args(%arg13 = %arg11) -> (tensor<1x768xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map(%arg8)
              %extracted_slice_410 = tensor.extract_slice %19[%arg8, %arg12] [%83, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_411 = tensor.extract_slice %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %85 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_410 : tensor<?x8xf32>) outs(%extracted_slice_411 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %86 = arith.mulf %in, %cst_142 : f32
                %87 = arith.addf %out, %86 : f32
                linalg.yield %87 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_412 = tensor.insert_slice %85 into %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
              scf.yield %inserted_slice_412 : tensor<1x768xf32>
            }
            scf.yield %82 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %26 = tensor.empty() : tensor<1x768xf32>
        %c0_242 = arith.constant 0 : index
        %c1_243 = arith.constant 1 : index
        %c8_244 = arith.constant 8 : index
        %c0_245 = arith.constant 0 : index
        %c768_246 = arith.constant 768 : index
        %c8_247 = arith.constant 8 : index
        %27 = scf.for %arg8 = %c0_242 to %c1_243 step %c8_244 iter_args(%arg9 = %26) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_245 to %c768_246 step %c8_247 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %83 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_410 : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_160 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_411 = tensor.insert_slice %83 into %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_411 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %c0_248 = arith.constant 0 : index
        %c1_249 = arith.constant 1 : index
        %c8_250 = arith.constant 8 : index
        %c0_251 = arith.constant 0 : index
        %c768_252 = arith.constant 768 : index
        %c8_253 = arith.constant 8 : index
        %c0_254 = arith.constant 0 : index
        %c768_255 = arith.constant 768 : index
        %c8_256 = arith.constant 8 : index
        %28 = scf.for %arg8 = %c0_248 to %c1_249 step %c8_250 iter_args(%arg9 = %27) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_251 to %c768_252 step %c8_253 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = scf.for %arg12 = %c0_254 to %c768_255 step %c8_256 iter_args(%arg13 = %arg11) -> (tensor<1x768xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map(%arg8)
              %extracted_slice_410 = tensor.extract_slice %19[%arg8, %arg12] [%83, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_411 = tensor.extract_slice %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %85 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_410 : tensor<?x8xf32>) outs(%extracted_slice_411 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %86 = arith.mulf %in, %cst_141 : f32
                %87 = arith.addf %out, %86 : f32
                linalg.yield %87 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_412 = tensor.insert_slice %85 into %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
              scf.yield %inserted_slice_412 : tensor<1x768xf32>
            }
            scf.yield %82 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %reshape = tensor.reshape %22(%cst_149) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x12x64xf32>
        %29 = tensor.empty() : tensor<32xf32>
        %30 = tensor.empty() : tensor<32xf32>
        %31 = arith.uitofp %arg1 : i64 to f32
        %c0_257 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c8_258 = arith.constant 8 : index
        %32:2 = scf.for %arg8 = %c0_257 to %c32 step %c8_258 iter_args(%arg9 = %29, %arg10 = %30) -> (tensor<32xf32>, tensor<32xf32>) {
          %extracted_slice_410 = tensor.extract_slice %arg9[%arg8] [8] [1] : tensor<32xf32> to tensor<8xf32>
          %extracted_slice_411 = tensor.extract_slice %arg10[%arg8] [8] [1] : tensor<32xf32> to tensor<8xf32>
          %81:2 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} outs(%extracted_slice_410, %extracted_slice_411 : tensor<8xf32>, tensor<8xf32>) {
          ^bb0(%out: f32, %out_414: f32):
            %82 = linalg.index 0 : index
            %83 = affine.apply #map6(%82, %arg8)
            %84 = arith.index_cast %83 : index to i64
            %85 = arith.uitofp %84 : i64 to f32
            %86 = arith.mulf %85, %cst_155 : f32
            %87 = arith.divf %86, %cst_156 : f32
            %88 = math.powf %cst_157, %87 : f32
            %89 = arith.mulf %31, %88 : f32
            %90 = math.cos %89 : f32
            %91 = math.sin %89 : f32
            linalg.yield %90, %91 : f32, f32
          } -> (tensor<8xf32>, tensor<8xf32>)
          %inserted_slice_412 = tensor.insert_slice %81#0 into %arg9[%arg8] [8] [1] : tensor<8xf32> into tensor<32xf32>
          %inserted_slice_413 = tensor.insert_slice %81#1 into %arg10[%arg8] [8] [1] : tensor<8xf32> into tensor<32xf32>
          scf.yield %inserted_slice_412, %inserted_slice_413 : tensor<32xf32>, tensor<32xf32>
        }
        %expanded = tensor.expand_shape %reshape [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : tensor<1x12x64xf32> into tensor<1x12x32x2xf32>
        %extracted_slice = tensor.extract_slice %expanded[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %extracted_slice_259 = tensor.extract_slice %expanded[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %expanded_260 = tensor.expand_shape %32#0 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %expanded_261 = tensor.expand_shape %32#1 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %33 = tensor.empty() : tensor<1x12x32x1xf32>
        %34 = tensor.empty() : tensor<1x12x32x1xf32>
        %c0_262 = arith.constant 0 : index
        %c1_263 = arith.constant 1 : index
        %c8_264 = arith.constant 8 : index
        %c0_265 = arith.constant 0 : index
        %c12_266 = arith.constant 12 : index
        %c8_267 = arith.constant 8 : index
        %c0_268 = arith.constant 0 : index
        %c32_269 = arith.constant 32 : index
        %c8_270 = arith.constant 8 : index
        %35:2 = scf.for %arg8 = %c0_262 to %c1_263 step %c8_264 iter_args(%arg9 = %33, %arg10 = %34) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
          %81:2 = scf.for %arg11 = %c0_265 to %c12_266 step %c8_267 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
            %82:2 = scf.for %arg14 = %c0_268 to %c32_269 step %c8_270 iter_args(%arg15 = %arg12, %arg16 = %arg13) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map7(%arg11)
              %85 = affine.min #map(%arg8)
              %86 = affine.min #map7(%arg11)
              %87 = affine.min #map(%arg8)
              %88 = affine.min #map7(%arg11)
              %89 = affine.min #map(%arg8)
              %90 = affine.min #map7(%arg11)
              %extracted_slice_410 = tensor.extract_slice %extracted_slice[%arg8, %arg11, %arg14, 0] [%83, %84, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_411 = tensor.extract_slice %extracted_slice_259[%arg8, %arg11, %arg14, 0] [%85, %86, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_412 = tensor.extract_slice %expanded_260[%arg14, 0] [8, 1] [1, 1] : tensor<32x1xf32> to tensor<8x1xf32>
              %extracted_slice_413 = tensor.extract_slice %expanded_261[%arg14, 0] [8, 1] [1, 1] : tensor<32x1xf32> to tensor<8x1xf32>
              %extracted_slice_414 = tensor.extract_slice %arg15[%arg8, %arg11, %arg14, 0] [%87, %88, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_415 = tensor.extract_slice %arg16[%arg8, %arg11, %arg14, 0] [%89, %90, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %91:2 = linalg.generic {indexing_maps = [#map8, #map8, #map9, #map9, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_410, %extracted_slice_411, %extracted_slice_412, %extracted_slice_413 : tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>, tensor<8x1xf32>, tensor<8x1xf32>) outs(%extracted_slice_414, %extracted_slice_415 : tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>) {
              ^bb0(%in: f32, %in_418: f32, %in_419: f32, %in_420: f32, %out: f32, %out_421: f32):
                %92 = arith.mulf %in, %in_419 : f32
                %93 = arith.mulf %in_418, %in_420 : f32
                %94 = arith.subf %92, %93 : f32
                %95 = arith.mulf %in_418, %in_419 : f32
                %96 = arith.mulf %in, %in_420 : f32
                %97 = arith.addf %95, %96 : f32
                linalg.yield %94, %97 : f32, f32
              } -> (tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>)
              %inserted_slice_416 = tensor.insert_slice %91#0 into %arg15[%arg8, %arg11, %arg14, 0] [%87, %88, 8, 1] [1, 1, 1, 1] : tensor<?x?x8x1xf32> into tensor<1x12x32x1xf32>
              %inserted_slice_417 = tensor.insert_slice %91#1 into %arg16[%arg8, %arg11, %arg14, 0] [%89, %90, 8, 1] [1, 1, 1, 1] : tensor<?x?x8x1xf32> into tensor<1x12x32x1xf32>
              scf.yield %inserted_slice_416, %inserted_slice_417 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
            }
            scf.yield %82#0, %82#1 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
          }
          scf.yield %81#0, %81#1 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
        }
        %36 = tensor.empty() : tensor<1x12x32x2xf32>
        %inserted_slice = tensor.insert_slice %35#0 into %36[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %inserted_slice_271 = tensor.insert_slice %35#1 into %inserted_slice[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %collapsed = tensor.collapse_shape %inserted_slice_271 [[0], [1], [2, 3]] : tensor<1x12x32x2xf32> into tensor<1x12x64xf32>
        %reshape_272 = tensor.reshape %collapsed(%cst_148) : (tensor<1x12x64xf32>, tensor<2xi64>) -> tensor<1x768xf32>
        %reshape_273 = tensor.reshape %25(%cst_149) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x12x64xf32>
        %37 = tensor.empty() : tensor<32xf32>
        %38 = tensor.empty() : tensor<32xf32>
        %39 = arith.uitofp %arg1 : i64 to f32
        %c0_274 = arith.constant 0 : index
        %c32_275 = arith.constant 32 : index
        %c8_276 = arith.constant 8 : index
        %40:2 = scf.for %arg8 = %c0_274 to %c32_275 step %c8_276 iter_args(%arg9 = %37, %arg10 = %38) -> (tensor<32xf32>, tensor<32xf32>) {
          %extracted_slice_410 = tensor.extract_slice %arg9[%arg8] [8] [1] : tensor<32xf32> to tensor<8xf32>
          %extracted_slice_411 = tensor.extract_slice %arg10[%arg8] [8] [1] : tensor<32xf32> to tensor<8xf32>
          %81:2 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} outs(%extracted_slice_410, %extracted_slice_411 : tensor<8xf32>, tensor<8xf32>) {
          ^bb0(%out: f32, %out_414: f32):
            %82 = linalg.index 0 : index
            %83 = affine.apply #map6(%82, %arg8)
            %84 = arith.index_cast %83 : index to i64
            %85 = arith.uitofp %84 : i64 to f32
            %86 = arith.mulf %85, %cst_155 : f32
            %87 = arith.divf %86, %cst_156 : f32
            %88 = math.powf %cst_157, %87 : f32
            %89 = arith.mulf %39, %88 : f32
            %90 = math.cos %89 : f32
            %91 = math.sin %89 : f32
            linalg.yield %90, %91 : f32, f32
          } -> (tensor<8xf32>, tensor<8xf32>)
          %inserted_slice_412 = tensor.insert_slice %81#0 into %arg9[%arg8] [8] [1] : tensor<8xf32> into tensor<32xf32>
          %inserted_slice_413 = tensor.insert_slice %81#1 into %arg10[%arg8] [8] [1] : tensor<8xf32> into tensor<32xf32>
          scf.yield %inserted_slice_412, %inserted_slice_413 : tensor<32xf32>, tensor<32xf32>
        }
        %expanded_277 = tensor.expand_shape %reshape_273 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : tensor<1x12x64xf32> into tensor<1x12x32x2xf32>
        %extracted_slice_278 = tensor.extract_slice %expanded_277[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %extracted_slice_279 = tensor.extract_slice %expanded_277[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %expanded_280 = tensor.expand_shape %40#0 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %expanded_281 = tensor.expand_shape %40#1 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %41 = tensor.empty() : tensor<1x12x32x1xf32>
        %42 = tensor.empty() : tensor<1x12x32x1xf32>
        %c0_282 = arith.constant 0 : index
        %c1_283 = arith.constant 1 : index
        %c8_284 = arith.constant 8 : index
        %c0_285 = arith.constant 0 : index
        %c12_286 = arith.constant 12 : index
        %c8_287 = arith.constant 8 : index
        %c0_288 = arith.constant 0 : index
        %c32_289 = arith.constant 32 : index
        %c8_290 = arith.constant 8 : index
        %43:2 = scf.for %arg8 = %c0_282 to %c1_283 step %c8_284 iter_args(%arg9 = %41, %arg10 = %42) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
          %81:2 = scf.for %arg11 = %c0_285 to %c12_286 step %c8_287 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
            %82:2 = scf.for %arg14 = %c0_288 to %c32_289 step %c8_290 iter_args(%arg15 = %arg12, %arg16 = %arg13) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map7(%arg11)
              %85 = affine.min #map(%arg8)
              %86 = affine.min #map7(%arg11)
              %87 = affine.min #map(%arg8)
              %88 = affine.min #map7(%arg11)
              %89 = affine.min #map(%arg8)
              %90 = affine.min #map7(%arg11)
              %extracted_slice_410 = tensor.extract_slice %extracted_slice_278[%arg8, %arg11, %arg14, 0] [%83, %84, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_411 = tensor.extract_slice %extracted_slice_279[%arg8, %arg11, %arg14, 0] [%85, %86, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_412 = tensor.extract_slice %expanded_280[%arg14, 0] [8, 1] [1, 1] : tensor<32x1xf32> to tensor<8x1xf32>
              %extracted_slice_413 = tensor.extract_slice %expanded_281[%arg14, 0] [8, 1] [1, 1] : tensor<32x1xf32> to tensor<8x1xf32>
              %extracted_slice_414 = tensor.extract_slice %arg15[%arg8, %arg11, %arg14, 0] [%87, %88, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_415 = tensor.extract_slice %arg16[%arg8, %arg11, %arg14, 0] [%89, %90, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %91:2 = linalg.generic {indexing_maps = [#map8, #map8, #map9, #map9, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_410, %extracted_slice_411, %extracted_slice_412, %extracted_slice_413 : tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>, tensor<8x1xf32>, tensor<8x1xf32>) outs(%extracted_slice_414, %extracted_slice_415 : tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>) {
              ^bb0(%in: f32, %in_418: f32, %in_419: f32, %in_420: f32, %out: f32, %out_421: f32):
                %92 = arith.mulf %in, %in_419 : f32
                %93 = arith.mulf %in_418, %in_420 : f32
                %94 = arith.subf %92, %93 : f32
                %95 = arith.mulf %in_418, %in_419 : f32
                %96 = arith.mulf %in, %in_420 : f32
                %97 = arith.addf %95, %96 : f32
                linalg.yield %94, %97 : f32, f32
              } -> (tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>)
              %inserted_slice_416 = tensor.insert_slice %91#0 into %arg15[%arg8, %arg11, %arg14, 0] [%87, %88, 8, 1] [1, 1, 1, 1] : tensor<?x?x8x1xf32> into tensor<1x12x32x1xf32>
              %inserted_slice_417 = tensor.insert_slice %91#1 into %arg16[%arg8, %arg11, %arg14, 0] [%89, %90, 8, 1] [1, 1, 1, 1] : tensor<?x?x8x1xf32> into tensor<1x12x32x1xf32>
              scf.yield %inserted_slice_416, %inserted_slice_417 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
            }
            scf.yield %82#0, %82#1 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
          }
          scf.yield %81#0, %81#1 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
        }
        %44 = tensor.empty() : tensor<1x12x32x2xf32>
        %inserted_slice_291 = tensor.insert_slice %43#0 into %44[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %inserted_slice_292 = tensor.insert_slice %43#1 into %inserted_slice_291[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %collapsed_293 = tensor.collapse_shape %inserted_slice_292 [[0], [1], [2, 3]] : tensor<1x12x32x2xf32> into tensor<1x12x64xf32>
        %reshape_294 = tensor.reshape %28(%cst_149) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x12x64xf32>
        %45 = tensor.empty() : tensor<32xf32>
        %46 = tensor.empty() : tensor<32xf32>
        %47 = arith.uitofp %arg1 : i64 to f32
        %c0_295 = arith.constant 0 : index
        %c32_296 = arith.constant 32 : index
        %c8_297 = arith.constant 8 : index
        %48:2 = scf.for %arg8 = %c0_295 to %c32_296 step %c8_297 iter_args(%arg9 = %45, %arg10 = %46) -> (tensor<32xf32>, tensor<32xf32>) {
          %extracted_slice_410 = tensor.extract_slice %arg9[%arg8] [8] [1] : tensor<32xf32> to tensor<8xf32>
          %extracted_slice_411 = tensor.extract_slice %arg10[%arg8] [8] [1] : tensor<32xf32> to tensor<8xf32>
          %81:2 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} outs(%extracted_slice_410, %extracted_slice_411 : tensor<8xf32>, tensor<8xf32>) {
          ^bb0(%out: f32, %out_414: f32):
            %82 = linalg.index 0 : index
            %83 = affine.apply #map6(%82, %arg8)
            %84 = arith.index_cast %83 : index to i64
            %85 = arith.uitofp %84 : i64 to f32
            %86 = arith.mulf %85, %cst_155 : f32
            %87 = arith.divf %86, %cst_156 : f32
            %88 = math.powf %cst_157, %87 : f32
            %89 = arith.mulf %47, %88 : f32
            %90 = math.cos %89 : f32
            %91 = math.sin %89 : f32
            linalg.yield %90, %91 : f32, f32
          } -> (tensor<8xf32>, tensor<8xf32>)
          %inserted_slice_412 = tensor.insert_slice %81#0 into %arg9[%arg8] [8] [1] : tensor<8xf32> into tensor<32xf32>
          %inserted_slice_413 = tensor.insert_slice %81#1 into %arg10[%arg8] [8] [1] : tensor<8xf32> into tensor<32xf32>
          scf.yield %inserted_slice_412, %inserted_slice_413 : tensor<32xf32>, tensor<32xf32>
        }
        %expanded_298 = tensor.expand_shape %reshape_294 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : tensor<1x12x64xf32> into tensor<1x12x32x2xf32>
        %extracted_slice_299 = tensor.extract_slice %expanded_298[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %extracted_slice_300 = tensor.extract_slice %expanded_298[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x2xf32> to tensor<1x12x32x1xf32>
        %expanded_301 = tensor.expand_shape %48#0 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %expanded_302 = tensor.expand_shape %48#1 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %49 = tensor.empty() : tensor<1x12x32x1xf32>
        %50 = tensor.empty() : tensor<1x12x32x1xf32>
        %c0_303 = arith.constant 0 : index
        %c1_304 = arith.constant 1 : index
        %c8_305 = arith.constant 8 : index
        %c0_306 = arith.constant 0 : index
        %c12_307 = arith.constant 12 : index
        %c8_308 = arith.constant 8 : index
        %c0_309 = arith.constant 0 : index
        %c32_310 = arith.constant 32 : index
        %c8_311 = arith.constant 8 : index
        %51:2 = scf.for %arg8 = %c0_303 to %c1_304 step %c8_305 iter_args(%arg9 = %49, %arg10 = %50) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
          %81:2 = scf.for %arg11 = %c0_306 to %c12_307 step %c8_308 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
            %82:2 = scf.for %arg14 = %c0_309 to %c32_310 step %c8_311 iter_args(%arg15 = %arg12, %arg16 = %arg13) -> (tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map7(%arg11)
              %85 = affine.min #map(%arg8)
              %86 = affine.min #map7(%arg11)
              %87 = affine.min #map(%arg8)
              %88 = affine.min #map7(%arg11)
              %89 = affine.min #map(%arg8)
              %90 = affine.min #map7(%arg11)
              %extracted_slice_410 = tensor.extract_slice %extracted_slice_299[%arg8, %arg11, %arg14, 0] [%83, %84, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_411 = tensor.extract_slice %extracted_slice_300[%arg8, %arg11, %arg14, 0] [%85, %86, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_412 = tensor.extract_slice %expanded_301[%arg14, 0] [8, 1] [1, 1] : tensor<32x1xf32> to tensor<8x1xf32>
              %extracted_slice_413 = tensor.extract_slice %expanded_302[%arg14, 0] [8, 1] [1, 1] : tensor<32x1xf32> to tensor<8x1xf32>
              %extracted_slice_414 = tensor.extract_slice %arg15[%arg8, %arg11, %arg14, 0] [%87, %88, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %extracted_slice_415 = tensor.extract_slice %arg16[%arg8, %arg11, %arg14, 0] [%89, %90, 8, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> to tensor<?x?x8x1xf32>
              %91:2 = linalg.generic {indexing_maps = [#map8, #map8, #map9, #map9, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_410, %extracted_slice_411, %extracted_slice_412, %extracted_slice_413 : tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>, tensor<8x1xf32>, tensor<8x1xf32>) outs(%extracted_slice_414, %extracted_slice_415 : tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>) {
              ^bb0(%in: f32, %in_418: f32, %in_419: f32, %in_420: f32, %out: f32, %out_421: f32):
                %92 = arith.mulf %in, %in_419 : f32
                %93 = arith.mulf %in_418, %in_420 : f32
                %94 = arith.subf %92, %93 : f32
                %95 = arith.mulf %in_418, %in_419 : f32
                %96 = arith.mulf %in, %in_420 : f32
                %97 = arith.addf %95, %96 : f32
                linalg.yield %94, %97 : f32, f32
              } -> (tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>)
              %inserted_slice_416 = tensor.insert_slice %91#0 into %arg15[%arg8, %arg11, %arg14, 0] [%87, %88, 8, 1] [1, 1, 1, 1] : tensor<?x?x8x1xf32> into tensor<1x12x32x1xf32>
              %inserted_slice_417 = tensor.insert_slice %91#1 into %arg16[%arg8, %arg11, %arg14, 0] [%89, %90, 8, 1] [1, 1, 1, 1] : tensor<?x?x8x1xf32> into tensor<1x12x32x1xf32>
              scf.yield %inserted_slice_416, %inserted_slice_417 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
            }
            scf.yield %82#0, %82#1 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
          }
          scf.yield %81#0, %81#1 : tensor<1x12x32x1xf32>, tensor<1x12x32x1xf32>
        }
        %52 = tensor.empty() : tensor<1x12x32x2xf32>
        %inserted_slice_312 = tensor.insert_slice %51#0 into %52[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %inserted_slice_313 = tensor.insert_slice %51#1 into %inserted_slice_312[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : tensor<1x12x32x1xf32> into tensor<1x12x32x2xf32>
        %collapsed_314 = tensor.collapse_shape %inserted_slice_313 [[0], [1], [2, 3]] : tensor<1x12x32x2xf32> into tensor<1x12x64xf32>
        %reshape_315 = tensor.reshape %collapsed_293(%cst_147) : (tensor<1x12x64xf32>, tensor<3xi64>) -> tensor<1x1x768xf32>
        %53 = arith.index_cast %14 : i64 to index
        %54 = arith.index_cast %arg1 : i64 to index
        %inserted_slice_316 = tensor.insert_slice %reshape_315 into %arg6[%53, %54, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
        %reshape_317 = tensor.reshape %collapsed_314(%cst_147) : (tensor<1x12x64xf32>, tensor<3xi64>) -> tensor<1x1x768xf32>
        %55 = arith.index_cast %14 : i64 to index
        %56 = arith.index_cast %arg1 : i64 to index
        %inserted_slice_318 = tensor.insert_slice %reshape_317 into %arg7[%55, %56, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
        %57 = arith.index_cast %14 : i64 to index
        %extracted_slice_319 = tensor.extract_slice %inserted_slice_316[%57, %c0, %c0] [1, 1024, 768] [1, 1, 1] : tensor<12x1024x768xf32> to tensor<1x1024x768xf32>
        %reshape_320 = tensor.reshape %extracted_slice_319(%cst_146) : (tensor<1x1024x768xf32>, tensor<2xi64>) -> tensor<1024x768xf32>
        %58 = arith.index_cast %14 : i64 to index
        %extracted_slice_321 = tensor.extract_slice %inserted_slice_318[%58, %c0, %c0] [1, 1024, 768] [1, 1, 1] : tensor<12x1024x768xf32> to tensor<1x1024x768xf32>
        %reshape_322 = tensor.reshape %extracted_slice_321(%cst_146) : (tensor<1x1024x768xf32>, tensor<2xi64>) -> tensor<1024x768xf32>
        %59 = scf.for %arg8 = %c0 to %c12 step %c1 iter_args(%arg9 = %cst_154) -> (tensor<1x12x64xf32>) {
          %81 = arith.index_cast %arg8 : index to i64
          %82 = arith.muli %81, %c64_i64 : i64
          %83 = arith.index_cast %82 : i64 to index
          %extracted_slice_410 = tensor.extract_slice %reshape_272[%c0, %83] [1, 64] [1, 1] : tensor<1x768xf32> to tensor<1x64xf32>
          %84 = arith.index_cast %82 : i64 to index
          %extracted_slice_411 = tensor.extract_slice %reshape_320[%c0, %84] [1024, 64] [1, 1] : tensor<1024x768xf32> to tensor<1024x64xf32>
          %85 = tensor.empty() : tensor<1x1024xf32>
          %c0_412 = arith.constant 0 : index
          %c1_413 = arith.constant 1 : index
          %c8_414 = arith.constant 8 : index
          %c0_415 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c8_416 = arith.constant 8 : index
          %86 = scf.for %arg10 = %c0_412 to %c1_413 step %c8_414 iter_args(%arg11 = %85) -> (tensor<1x1024xf32>) {
            %103 = scf.for %arg12 = %c0_415 to %c1024 step %c8_416 iter_args(%arg13 = %arg11) -> (tensor<1x1024xf32>) {
              %104 = affine.min #map(%arg10)
              %extracted_slice_473 = tensor.extract_slice %arg13[%arg10, %arg12] [%104, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
              %105 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_473 : tensor<?x8xf32>) {
              ^bb0(%out: f32):
                linalg.yield %cst_160 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_474 = tensor.insert_slice %105 into %arg13[%arg10, %arg12] [%104, 8] [1, 1] : tensor<?x8xf32> into tensor<1x1024xf32>
              scf.yield %inserted_slice_474 : tensor<1x1024xf32>
            }
            scf.yield %103 : tensor<1x1024xf32>
          }
          %c0_417 = arith.constant 0 : index
          %c1_418 = arith.constant 1 : index
          %c8_419 = arith.constant 8 : index
          %c0_420 = arith.constant 0 : index
          %c1024_421 = arith.constant 1024 : index
          %c8_422 = arith.constant 8 : index
          %c0_423 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c8_424 = arith.constant 8 : index
          %87 = scf.for %arg10 = %c0_417 to %c1_418 step %c8_419 iter_args(%arg11 = %86) -> (tensor<1x1024xf32>) {
            %103 = scf.for %arg12 = %c0_420 to %c1024_421 step %c8_422 iter_args(%arg13 = %arg11) -> (tensor<1x1024xf32>) {
              %104 = scf.for %arg14 = %c0_423 to %c64 step %c8_424 iter_args(%arg15 = %arg13) -> (tensor<1x1024xf32>) {
                %105 = affine.min #map(%arg10)
                %106 = affine.min #map(%arg10)
                %extracted_slice_473 = tensor.extract_slice %extracted_slice_410[%arg10, %arg14] [%105, 8] [1, 1] : tensor<1x64xf32> to tensor<?x8xf32>
                %extracted_slice_474 = tensor.extract_slice %extracted_slice_411[%arg12, %arg14] [8, 8] [1, 1] : tensor<1024x64xf32> to tensor<8x8xf32>
                %extracted_slice_475 = tensor.extract_slice %arg15[%arg10, %arg12] [%106, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
                %107 = linalg.generic {indexing_maps = [#map4, #map10, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_473, %extracted_slice_474 : tensor<?x8xf32>, tensor<8x8xf32>) outs(%extracted_slice_475 : tensor<?x8xf32>) {
                ^bb0(%in: f32, %in_477: f32, %out: f32):
                  %108 = linalg.index 1 : index
                  %109 = affine.apply #map6(%108, %arg12)
                  %110 = arith.index_cast %109 : index to i64
                  %111 = arith.cmpi sle, %110, %arg1 : i64
                  %112 = arith.select %111, %cst_161, %cst_160 : f32
                  %113 = arith.mulf %in, %in_477 : f32
                  %114 = arith.addf %out, %113 : f32
                  %115 = arith.cmpf ugt, %112, %cst_152 : f32
                  %116 = arith.select %115, %114, %cst_153 : f32
                  linalg.yield %116 : f32
                } -> tensor<?x8xf32>
                %inserted_slice_476 = tensor.insert_slice %107 into %arg15[%arg10, %arg12] [%106, 8] [1, 1] : tensor<?x8xf32> into tensor<1x1024xf32>
                scf.yield %inserted_slice_476 : tensor<1x1024xf32>
              }
              scf.yield %104 : tensor<1x1024xf32>
            }
            scf.yield %103 : tensor<1x1024xf32>
          }
          %88 = tensor.empty() : tensor<1x1024xf32>
          %c0_425 = arith.constant 0 : index
          %c1_426 = arith.constant 1 : index
          %c8_427 = arith.constant 8 : index
          %c0_428 = arith.constant 0 : index
          %c1024_429 = arith.constant 1024 : index
          %c8_430 = arith.constant 8 : index
          %89 = scf.for %arg10 = %c0_425 to %c1_426 step %c8_427 iter_args(%arg11 = %88) -> (tensor<1x1024xf32>) {
            %103 = scf.for %arg12 = %c0_428 to %c1024_429 step %c8_430 iter_args(%arg13 = %arg11) -> (tensor<1x1024xf32>) {
              %104 = affine.min #map(%arg10)
              %105 = affine.min #map(%arg10)
              %extracted_slice_473 = tensor.extract_slice %87[%arg10, %arg12] [%104, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
              %extracted_slice_474 = tensor.extract_slice %arg13[%arg10, %arg12] [%105, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
              %106 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_473 : tensor<?x8xf32>) outs(%extracted_slice_474 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %107 = arith.mulf %in, %cst_159 : f32
                linalg.yield %107 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_475 = tensor.insert_slice %106 into %arg13[%arg10, %arg12] [%105, 8] [1, 1] : tensor<?x8xf32> into tensor<1x1024xf32>
              scf.yield %inserted_slice_475 : tensor<1x1024xf32>
            }
            scf.yield %103 : tensor<1x1024xf32>
          }
          %90 = tensor.empty() : tensor<1xf32>
          %c0_431 = arith.constant 0 : index
          %c1_432 = arith.constant 1 : index
          %c8_433 = arith.constant 8 : index
          %91 = scf.for %arg10 = %c0_431 to %c1_432 step %c8_433 iter_args(%arg11 = %90) -> (tensor<1xf32>) {
            %103 = affine.min #map(%arg10)
            %extracted_slice_473 = tensor.extract_slice %arg11[%arg10] [%103] [1] : tensor<1xf32> to tensor<?xf32>
            %104 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_473 : tensor<?xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_140 : f32
            } -> tensor<?xf32>
            %inserted_slice_474 = tensor.insert_slice %104 into %arg11[%arg10] [%103] [1] : tensor<?xf32> into tensor<1xf32>
            scf.yield %inserted_slice_474 : tensor<1xf32>
          }
          %c0_434 = arith.constant 0 : index
          %c1_435 = arith.constant 1 : index
          %c8_436 = arith.constant 8 : index
          %c0_437 = arith.constant 0 : index
          %c1024_438 = arith.constant 1024 : index
          %c8_439 = arith.constant 8 : index
          %92 = scf.for %arg10 = %c0_434 to %c1_435 step %c8_436 iter_args(%arg11 = %91) -> (tensor<1xf32>) {
            %103 = scf.for %arg12 = %c0_437 to %c1024_438 step %c8_439 iter_args(%arg13 = %arg11) -> (tensor<1xf32>) {
              %104 = affine.min #map(%arg10)
              %105 = affine.min #map(%arg10)
              %extracted_slice_473 = tensor.extract_slice %89[%arg10, %arg12] [%104, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
              %extracted_slice_474 = tensor.extract_slice %arg13[%arg10] [%105] [1] : tensor<1xf32> to tensor<?xf32>
              %106 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_473 : tensor<?x8xf32>) outs(%extracted_slice_474 : tensor<?xf32>) {
              ^bb0(%in: f32, %out: f32):
                %107 = arith.maxnumf %in, %out : f32
                linalg.yield %107 : f32
              } -> tensor<?xf32>
              %inserted_slice_475 = tensor.insert_slice %106 into %arg13[%arg10] [%105] [1] : tensor<?xf32> into tensor<1xf32>
              scf.yield %inserted_slice_475 : tensor<1xf32>
            }
            scf.yield %103 : tensor<1xf32>
          }
          %93 = tensor.empty() : tensor<1x1024xf32>
          %c0_440 = arith.constant 0 : index
          %c1_441 = arith.constant 1 : index
          %c8_442 = arith.constant 8 : index
          %c0_443 = arith.constant 0 : index
          %c1024_444 = arith.constant 1024 : index
          %c8_445 = arith.constant 8 : index
          %94 = scf.for %arg10 = %c0_440 to %c1_441 step %c8_442 iter_args(%arg11 = %93) -> (tensor<1x1024xf32>) {
            %103 = scf.for %arg12 = %c0_443 to %c1024_444 step %c8_445 iter_args(%arg13 = %arg11) -> (tensor<1x1024xf32>) {
              %104 = affine.min #map(%arg10)
              %105 = affine.min #map(%arg10)
              %106 = affine.min #map(%arg10)
              %extracted_slice_473 = tensor.extract_slice %89[%arg10, %arg12] [%104, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
              %extracted_slice_474 = tensor.extract_slice %92[%arg10] [%105] [1] : tensor<1xf32> to tensor<?xf32>
              %extracted_slice_475 = tensor.extract_slice %arg13[%arg10, %arg12] [%106, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
              %107 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_473, %extracted_slice_474 : tensor<?x8xf32>, tensor<?xf32>) outs(%extracted_slice_475 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %in_477: f32, %out: f32):
                %108 = arith.subf %in, %in_477 : f32
                %109 = math.exp %108 : f32
                linalg.yield %109 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_476 = tensor.insert_slice %107 into %arg13[%arg10, %arg12] [%106, 8] [1, 1] : tensor<?x8xf32> into tensor<1x1024xf32>
              scf.yield %inserted_slice_476 : tensor<1x1024xf32>
            }
            scf.yield %103 : tensor<1x1024xf32>
          }
          %95 = tensor.empty() : tensor<1xf32>
          %c0_446 = arith.constant 0 : index
          %c1_447 = arith.constant 1 : index
          %c8_448 = arith.constant 8 : index
          %96 = scf.for %arg10 = %c0_446 to %c1_447 step %c8_448 iter_args(%arg11 = %95) -> (tensor<1xf32>) {
            %103 = affine.min #map(%arg10)
            %extracted_slice_473 = tensor.extract_slice %arg11[%arg10] [%103] [1] : tensor<1xf32> to tensor<?xf32>
            %104 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_473 : tensor<?xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_160 : f32
            } -> tensor<?xf32>
            %inserted_slice_474 = tensor.insert_slice %104 into %arg11[%arg10] [%103] [1] : tensor<?xf32> into tensor<1xf32>
            scf.yield %inserted_slice_474 : tensor<1xf32>
          }
          %c0_449 = arith.constant 0 : index
          %c1_450 = arith.constant 1 : index
          %c8_451 = arith.constant 8 : index
          %c0_452 = arith.constant 0 : index
          %c1024_453 = arith.constant 1024 : index
          %c8_454 = arith.constant 8 : index
          %97 = scf.for %arg10 = %c0_449 to %c1_450 step %c8_451 iter_args(%arg11 = %96) -> (tensor<1xf32>) {
            %103 = scf.for %arg12 = %c0_452 to %c1024_453 step %c8_454 iter_args(%arg13 = %arg11) -> (tensor<1xf32>) {
              %104 = affine.min #map(%arg10)
              %105 = affine.min #map(%arg10)
              %extracted_slice_473 = tensor.extract_slice %94[%arg10, %arg12] [%104, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
              %extracted_slice_474 = tensor.extract_slice %arg13[%arg10] [%105] [1] : tensor<1xf32> to tensor<?xf32>
              %106 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_473 : tensor<?x8xf32>) outs(%extracted_slice_474 : tensor<?xf32>) {
              ^bb0(%in: f32, %out: f32):
                %107 = arith.addf %in, %out : f32
                linalg.yield %107 : f32
              } -> tensor<?xf32>
              %inserted_slice_475 = tensor.insert_slice %106 into %arg13[%arg10] [%105] [1] : tensor<?xf32> into tensor<1xf32>
              scf.yield %inserted_slice_475 : tensor<1xf32>
            }
            scf.yield %103 : tensor<1xf32>
          }
          %98 = arith.index_cast %82 : i64 to index
          %extracted_slice_455 = tensor.extract_slice %reshape_322[%c0, %98] [1024, 64] [1, 1] : tensor<1024x768xf32> to tensor<1024x64xf32>
          %99 = tensor.empty() : tensor<1x64xf32>
          %c0_456 = arith.constant 0 : index
          %c1_457 = arith.constant 1 : index
          %c8_458 = arith.constant 8 : index
          %c0_459 = arith.constant 0 : index
          %c64_460 = arith.constant 64 : index
          %c8_461 = arith.constant 8 : index
          %100 = scf.for %arg10 = %c0_456 to %c1_457 step %c8_458 iter_args(%arg11 = %99) -> (tensor<1x64xf32>) {
            %103 = scf.for %arg12 = %c0_459 to %c64_460 step %c8_461 iter_args(%arg13 = %arg11) -> (tensor<1x64xf32>) {
              %104 = affine.min #map(%arg10)
              %extracted_slice_473 = tensor.extract_slice %arg13[%arg10, %arg12] [%104, 8] [1, 1] : tensor<1x64xf32> to tensor<?x8xf32>
              %105 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_473 : tensor<?x8xf32>) {
              ^bb0(%out: f32):
                linalg.yield %cst_160 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_474 = tensor.insert_slice %105 into %arg13[%arg10, %arg12] [%104, 8] [1, 1] : tensor<?x8xf32> into tensor<1x64xf32>
              scf.yield %inserted_slice_474 : tensor<1x64xf32>
            }
            scf.yield %103 : tensor<1x64xf32>
          }
          %c0_462 = arith.constant 0 : index
          %c1_463 = arith.constant 1 : index
          %c8_464 = arith.constant 8 : index
          %c0_465 = arith.constant 0 : index
          %c64_466 = arith.constant 64 : index
          %c8_467 = arith.constant 8 : index
          %c0_468 = arith.constant 0 : index
          %c1024_469 = arith.constant 1024 : index
          %c8_470 = arith.constant 8 : index
          %101 = scf.for %arg10 = %c0_462 to %c1_463 step %c8_464 iter_args(%arg11 = %100) -> (tensor<1x64xf32>) {
            %103 = scf.for %arg12 = %c0_465 to %c64_466 step %c8_467 iter_args(%arg13 = %arg11) -> (tensor<1x64xf32>) {
              %104 = scf.for %arg14 = %c0_468 to %c1024_469 step %c8_470 iter_args(%arg15 = %arg13) -> (tensor<1x64xf32>) {
                %105 = affine.min #map(%arg10)
                %106 = affine.min #map(%arg10)
                %107 = affine.min #map(%arg10)
                %extracted_slice_473 = tensor.extract_slice %94[%arg10, %arg14] [%105, 8] [1, 1] : tensor<1x1024xf32> to tensor<?x8xf32>
                %extracted_slice_474 = tensor.extract_slice %97[%arg10] [%106] [1] : tensor<1xf32> to tensor<?xf32>
                %extracted_slice_475 = tensor.extract_slice %extracted_slice_455[%arg14, %arg12] [8, 8] [1, 1] : tensor<1024x64xf32> to tensor<8x8xf32>
                %extracted_slice_476 = tensor.extract_slice %arg15[%arg10, %arg12] [%107, 8] [1, 1] : tensor<1x64xf32> to tensor<?x8xf32>
                %108 = linalg.generic {indexing_maps = [#map4, #map11, #map12, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_473, %extracted_slice_474, %extracted_slice_475 : tensor<?x8xf32>, tensor<?xf32>, tensor<8x8xf32>) outs(%extracted_slice_476 : tensor<?x8xf32>) {
                ^bb0(%in: f32, %in_478: f32, %in_479: f32, %out: f32):
                  %109 = arith.divf %in, %in_478 : f32
                  %110 = arith.mulf %109, %in_479 : f32
                  %111 = arith.addf %out, %110 : f32
                  linalg.yield %111 : f32
                } -> tensor<?x8xf32>
                %inserted_slice_477 = tensor.insert_slice %108 into %arg15[%arg10, %arg12] [%107, 8] [1, 1] : tensor<?x8xf32> into tensor<1x64xf32>
                scf.yield %inserted_slice_477 : tensor<1x64xf32>
              }
              scf.yield %104 : tensor<1x64xf32>
            }
            scf.yield %103 : tensor<1x64xf32>
          }
          %reshape_471 = tensor.reshape %101(%cst_145) : (tensor<1x64xf32>, tensor<3xi64>) -> tensor<1x1x64xf32>
          %102 = arith.index_cast %81 : i64 to index
          %inserted_slice_472 = tensor.insert_slice %reshape_471 into %arg9[%c0, %102, 0] [1, 1, 64] [1, 1, 1] : tensor<1x1x64xf32> into tensor<1x12x64xf32>
          scf.yield %inserted_slice_472 : tensor<1x12x64xf32>
        }
        %reshape_323 = tensor.reshape %59(%cst_148) : (tensor<1x12x64xf32>, tensor<2xi64>) -> tensor<1x768xf32>
        %60 = tensor.empty() : tensor<1x768xf32>
        %c0_324 = arith.constant 0 : index
        %c1_325 = arith.constant 1 : index
        %c8_326 = arith.constant 8 : index
        %c0_327 = arith.constant 0 : index
        %c768_328 = arith.constant 768 : index
        %c8_329 = arith.constant 8 : index
        %61 = scf.for %arg8 = %c0_324 to %c1_325 step %c8_326 iter_args(%arg9 = %60) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_327 to %c768_328 step %c8_329 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %83 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_410 : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_160 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_411 = tensor.insert_slice %83 into %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_411 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %c0_330 = arith.constant 0 : index
        %c1_331 = arith.constant 1 : index
        %c8_332 = arith.constant 8 : index
        %c0_333 = arith.constant 0 : index
        %c768_334 = arith.constant 768 : index
        %c8_335 = arith.constant 8 : index
        %c0_336 = arith.constant 0 : index
        %c768_337 = arith.constant 768 : index
        %c8_338 = arith.constant 8 : index
        %62 = scf.for %arg8 = %c0_330 to %c1_331 step %c8_332 iter_args(%arg9 = %61) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_333 to %c768_334 step %c8_335 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = scf.for %arg12 = %c0_336 to %c768_337 step %c8_338 iter_args(%arg13 = %arg11) -> (tensor<1x768xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map(%arg8)
              %extracted_slice_410 = tensor.extract_slice %reshape_323[%arg8, %arg12] [%83, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_411 = tensor.extract_slice %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %85 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_410 : tensor<?x8xf32>) outs(%extracted_slice_411 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %86 = arith.mulf %in, %cst_139 : f32
                %87 = arith.addf %out, %86 : f32
                linalg.yield %87 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_412 = tensor.insert_slice %85 into %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
              scf.yield %inserted_slice_412 : tensor<1x768xf32>
            }
            scf.yield %82 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %63 = tensor.empty() : tensor<1x768xf32>
        %c0_339 = arith.constant 0 : index
        %c1_340 = arith.constant 1 : index
        %c8_341 = arith.constant 8 : index
        %c0_342 = arith.constant 0 : index
        %c768_343 = arith.constant 768 : index
        %c8_344 = arith.constant 8 : index
        %64 = scf.for %arg8 = %c0_339 to %c1_340 step %c8_341 iter_args(%arg9 = %63) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_342 to %c768_343 step %c8_344 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %83 = affine.min #map(%arg8)
            %84 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg5[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_411 = tensor.extract_slice %62[%arg8, %arg10] [%83, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_412 = tensor.extract_slice %arg11[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %85 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_410, %extracted_slice_411 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%extracted_slice_412 : tensor<?x8xf32>) {
            ^bb0(%in: f32, %in_414: f32, %out: f32):
              %86 = arith.addf %in, %in_414 : f32
              linalg.yield %86 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_413 = tensor.insert_slice %85 into %arg11[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_413 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %65 = tensor.empty() : tensor<1xf32>
        %c0_345 = arith.constant 0 : index
        %c1_346 = arith.constant 1 : index
        %c8_347 = arith.constant 8 : index
        %66 = scf.for %arg8 = %c0_345 to %c1_346 step %c8_347 iter_args(%arg9 = %65) -> (tensor<1xf32>) {
          %81 = affine.min #map(%arg8)
          %extracted_slice_410 = tensor.extract_slice %arg9[%arg8] [%81] [1] : tensor<1xf32> to tensor<?xf32>
          %82 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_410 : tensor<?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_160 : f32
          } -> tensor<?xf32>
          %inserted_slice_411 = tensor.insert_slice %82 into %arg9[%arg8] [%81] [1] : tensor<?xf32> into tensor<1xf32>
          scf.yield %inserted_slice_411 : tensor<1xf32>
        }
        %c0_348 = arith.constant 0 : index
        %c1_349 = arith.constant 1 : index
        %c8_350 = arith.constant 8 : index
        %c0_351 = arith.constant 0 : index
        %c768_352 = arith.constant 768 : index
        %c8_353 = arith.constant 8 : index
        %67 = scf.for %arg8 = %c0_348 to %c1_349 step %c8_350 iter_args(%arg9 = %66) -> (tensor<1xf32>) {
          %81 = scf.for %arg10 = %c0_351 to %c768_352 step %c8_353 iter_args(%arg11 = %arg9) -> (tensor<1xf32>) {
            %82 = affine.min #map(%arg8)
            %83 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %64[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_411 = tensor.extract_slice %arg11[%arg8] [%83] [1] : tensor<1xf32> to tensor<?xf32>
            %84 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_410 : tensor<?x8xf32>) outs(%extracted_slice_411 : tensor<?xf32>) {
            ^bb0(%in: f32, %out: f32):
              %85 = arith.mulf %in, %in : f32
              %86 = arith.addf %out, %85 : f32
              linalg.yield %86 : f32
            } -> tensor<?xf32>
            %inserted_slice_412 = tensor.insert_slice %84 into %arg11[%arg8] [%83] [1] : tensor<?xf32> into tensor<1xf32>
            scf.yield %inserted_slice_412 : tensor<1xf32>
          }
          scf.yield %81 : tensor<1xf32>
        }
        %68 = tensor.empty() : tensor<1x768xf32>
        %c0_354 = arith.constant 0 : index
        %c1_355 = arith.constant 1 : index
        %c8_356 = arith.constant 8 : index
        %c0_357 = arith.constant 0 : index
        %c768_358 = arith.constant 768 : index
        %c8_359 = arith.constant 8 : index
        %69 = scf.for %arg8 = %c0_354 to %c1_355 step %c8_356 iter_args(%arg9 = %68) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_357 to %c768_358 step %c8_359 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %83 = affine.min #map(%arg8)
            %84 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %64[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_411 = tensor.extract_slice %67[%arg8] [%83] [1] : tensor<1xf32> to tensor<?xf32>
            %extracted_slice_412 = tensor.extract_slice %arg11[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %85 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_410, %extracted_slice_411 : tensor<?x8xf32>, tensor<?xf32>) outs(%extracted_slice_412 : tensor<?x8xf32>) {
            ^bb0(%in: f32, %in_414: f32, %out: f32):
              %86 = arith.divf %in_414, %cst_150 : f32
              %87 = arith.addf %86, %cst_158 : f32
              %88 = math.rsqrt %87 : f32
              %89 = arith.mulf %in, %88 : f32
              %90 = arith.mulf %89, %cst_138 : f32
              linalg.yield %90 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_413 = tensor.insert_slice %85 into %arg11[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_413 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %70 = tensor.empty() : tensor<1x2048xf32>
        %c0_360 = arith.constant 0 : index
        %c1_361 = arith.constant 1 : index
        %c8_362 = arith.constant 8 : index
        %c0_363 = arith.constant 0 : index
        %c2048 = arith.constant 2048 : index
        %c8_364 = arith.constant 8 : index
        %71 = scf.for %arg8 = %c0_360 to %c1_361 step %c8_362 iter_args(%arg9 = %70) -> (tensor<1x2048xf32>) {
          %81 = scf.for %arg10 = %c0_363 to %c2048 step %c8_364 iter_args(%arg11 = %arg9) -> (tensor<1x2048xf32>) {
            %82 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x2048xf32> to tensor<?x8xf32>
            %83 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_410 : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_160 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_411 = tensor.insert_slice %83 into %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<?x8xf32> into tensor<1x2048xf32>
            scf.yield %inserted_slice_411 : tensor<1x2048xf32>
          }
          scf.yield %81 : tensor<1x2048xf32>
        }
        %c0_365 = arith.constant 0 : index
        %c1_366 = arith.constant 1 : index
        %c8_367 = arith.constant 8 : index
        %c0_368 = arith.constant 0 : index
        %c2048_369 = arith.constant 2048 : index
        %c8_370 = arith.constant 8 : index
        %c0_371 = arith.constant 0 : index
        %c768_372 = arith.constant 768 : index
        %c8_373 = arith.constant 8 : index
        %72 = scf.for %arg8 = %c0_365 to %c1_366 step %c8_367 iter_args(%arg9 = %71) -> (tensor<1x2048xf32>) {
          %81 = scf.for %arg10 = %c0_368 to %c2048_369 step %c8_370 iter_args(%arg11 = %arg9) -> (tensor<1x2048xf32>) {
            %82 = scf.for %arg12 = %c0_371 to %c768_372 step %c8_373 iter_args(%arg13 = %arg11) -> (tensor<1x2048xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map(%arg8)
              %extracted_slice_410 = tensor.extract_slice %69[%arg8, %arg12] [%83, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_411 = tensor.extract_slice %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x2048xf32> to tensor<?x8xf32>
              %85 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_410 : tensor<?x8xf32>) outs(%extracted_slice_411 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %86 = arith.mulf %in, %cst_137 : f32
                %87 = arith.addf %out, %86 : f32
                linalg.yield %87 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_412 = tensor.insert_slice %85 into %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x2048xf32>
              scf.yield %inserted_slice_412 : tensor<1x2048xf32>
            }
            scf.yield %82 : tensor<1x2048xf32>
          }
          scf.yield %81 : tensor<1x2048xf32>
        }
        %73 = tensor.empty() : tensor<1x2048xf32>
        %c0_374 = arith.constant 0 : index
        %c1_375 = arith.constant 1 : index
        %c8_376 = arith.constant 8 : index
        %c0_377 = arith.constant 0 : index
        %c2048_378 = arith.constant 2048 : index
        %c8_379 = arith.constant 8 : index
        %74 = scf.for %arg8 = %c0_374 to %c1_375 step %c8_376 iter_args(%arg9 = %73) -> (tensor<1x2048xf32>) {
          %81 = scf.for %arg10 = %c0_377 to %c2048_378 step %c8_379 iter_args(%arg11 = %arg9) -> (tensor<1x2048xf32>) {
            %82 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x2048xf32> to tensor<?x8xf32>
            %83 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_410 : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_160 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_411 = tensor.insert_slice %83 into %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<?x8xf32> into tensor<1x2048xf32>
            scf.yield %inserted_slice_411 : tensor<1x2048xf32>
          }
          scf.yield %81 : tensor<1x2048xf32>
        }
        %c0_380 = arith.constant 0 : index
        %c1_381 = arith.constant 1 : index
        %c8_382 = arith.constant 8 : index
        %c0_383 = arith.constant 0 : index
        %c2048_384 = arith.constant 2048 : index
        %c8_385 = arith.constant 8 : index
        %c0_386 = arith.constant 0 : index
        %c768_387 = arith.constant 768 : index
        %c8_388 = arith.constant 8 : index
        %75 = scf.for %arg8 = %c0_380 to %c1_381 step %c8_382 iter_args(%arg9 = %74) -> (tensor<1x2048xf32>) {
          %81 = scf.for %arg10 = %c0_383 to %c2048_384 step %c8_385 iter_args(%arg11 = %arg9) -> (tensor<1x2048xf32>) {
            %82 = scf.for %arg12 = %c0_386 to %c768_387 step %c8_388 iter_args(%arg13 = %arg11) -> (tensor<1x2048xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map(%arg8)
              %extracted_slice_410 = tensor.extract_slice %69[%arg8, %arg12] [%83, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_411 = tensor.extract_slice %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x2048xf32> to tensor<?x8xf32>
              %85 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_410 : tensor<?x8xf32>) outs(%extracted_slice_411 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %86 = arith.mulf %in, %cst_136 : f32
                %87 = arith.addf %out, %86 : f32
                linalg.yield %87 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_412 = tensor.insert_slice %85 into %arg13[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x2048xf32>
              scf.yield %inserted_slice_412 : tensor<1x2048xf32>
            }
            scf.yield %82 : tensor<1x2048xf32>
          }
          scf.yield %81 : tensor<1x2048xf32>
        }
        %76 = tensor.empty() : tensor<1x768xf32>
        %c0_389 = arith.constant 0 : index
        %c1_390 = arith.constant 1 : index
        %c8_391 = arith.constant 8 : index
        %c0_392 = arith.constant 0 : index
        %c768_393 = arith.constant 768 : index
        %c8_394 = arith.constant 8 : index
        %77 = scf.for %arg8 = %c0_389 to %c1_390 step %c8_391 iter_args(%arg9 = %76) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_392 to %c768_393 step %c8_394 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %83 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice_410 : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_160 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_411 = tensor.insert_slice %83 into %arg11[%arg8, %arg10] [%82, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_411 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %c0_395 = arith.constant 0 : index
        %c1_396 = arith.constant 1 : index
        %c8_397 = arith.constant 8 : index
        %c0_398 = arith.constant 0 : index
        %c768_399 = arith.constant 768 : index
        %c8_400 = arith.constant 8 : index
        %c0_401 = arith.constant 0 : index
        %c2048_402 = arith.constant 2048 : index
        %c8_403 = arith.constant 8 : index
        %78 = scf.for %arg8 = %c0_395 to %c1_396 step %c8_397 iter_args(%arg9 = %77) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_398 to %c768_399 step %c8_400 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = scf.for %arg12 = %c0_401 to %c2048_402 step %c8_403 iter_args(%arg13 = %arg11) -> (tensor<1x768xf32>) {
              %83 = affine.min #map(%arg8)
              %84 = affine.min #map(%arg8)
              %85 = affine.min #map(%arg8)
              %extracted_slice_410 = tensor.extract_slice %72[%arg8, %arg12] [%83, 8] [1, 1] : tensor<1x2048xf32> to tensor<?x8xf32>
              %extracted_slice_411 = tensor.extract_slice %75[%arg8, %arg12] [%84, 8] [1, 1] : tensor<1x2048xf32> to tensor<?x8xf32>
              %extracted_slice_412 = tensor.extract_slice %arg13[%arg8, %arg10] [%85, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %86 = linalg.generic {indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_410, %extracted_slice_411 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%extracted_slice_412 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %in_414: f32, %out: f32):
                %87 = arith.negf %in : f32
                %88 = math.exp %87 : f32
                %89 = arith.addf %in, %88 : f32
                %90 = arith.divf %in, %89 : f32
                %91 = arith.mulf %90, %in_414 : f32
                %92 = arith.mulf %91, %cst_135 : f32
                %93 = arith.addf %out, %92 : f32
                linalg.yield %93 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_413 = tensor.insert_slice %86 into %arg13[%arg8, %arg10] [%85, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
              scf.yield %inserted_slice_413 : tensor<1x768xf32>
            }
            scf.yield %82 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        %79 = tensor.empty() : tensor<1x768xf32>
        %c0_404 = arith.constant 0 : index
        %c1_405 = arith.constant 1 : index
        %c8_406 = arith.constant 8 : index
        %c0_407 = arith.constant 0 : index
        %c768_408 = arith.constant 768 : index
        %c8_409 = arith.constant 8 : index
        %80 = scf.for %arg8 = %c0_404 to %c1_405 step %c8_406 iter_args(%arg9 = %79) -> (tensor<1x768xf32>) {
          %81 = scf.for %arg10 = %c0_407 to %c768_408 step %c8_409 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %82 = affine.min #map(%arg8)
            %83 = affine.min #map(%arg8)
            %84 = affine.min #map(%arg8)
            %extracted_slice_410 = tensor.extract_slice %64[%arg8, %arg10] [%82, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_411 = tensor.extract_slice %78[%arg8, %arg10] [%83, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_412 = tensor.extract_slice %arg11[%arg8, %arg10] [%84, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %85 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_410, %extracted_slice_411 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%extracted_slice_412 : tensor<?x8xf32>) {
            ^bb0(%in: f32, %in_414: f32, %out: f32):
              %86 = arith.addf %in, %in_414 : f32
              linalg.yield %86 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_413 = tensor.insert_slice %85 into %arg11[%arg8, %arg10] [%84, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_413 : tensor<1x768xf32>
          }
          scf.yield %81 : tensor<1x768xf32>
        }
        scf.yield %80, %inserted_slice_316, %inserted_slice_318 : tensor<1x768xf32>, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
      }
      %2 = tensor.empty() : tensor<1xf32>
      %c0_163 = arith.constant 0 : index
      %c1_164 = arith.constant 1 : index
      %c8_165 = arith.constant 8 : index
      %3 = scf.for %arg4 = %c0_163 to %c1_164 step %c8_165 iter_args(%arg5 = %2) -> (tensor<1xf32>) {
        %14 = affine.min #map(%arg4)
        %extracted_slice = tensor.extract_slice %arg5[%arg4] [%14] [1] : tensor<1xf32> to tensor<?xf32>
        %15 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_160 : f32
        } -> tensor<?xf32>
        %inserted_slice = tensor.insert_slice %15 into %arg5[%arg4] [%14] [1] : tensor<?xf32> into tensor<1xf32>
        scf.yield %inserted_slice : tensor<1xf32>
      }
      %c0_166 = arith.constant 0 : index
      %c1_167 = arith.constant 1 : index
      %c8_168 = arith.constant 8 : index
      %c0_169 = arith.constant 0 : index
      %c768 = arith.constant 768 : index
      %c8_170 = arith.constant 8 : index
      %4 = scf.for %arg4 = %c0_166 to %c1_167 step %c8_168 iter_args(%arg5 = %3) -> (tensor<1xf32>) {
        %14 = scf.for %arg6 = %c0_169 to %c768 step %c8_170 iter_args(%arg7 = %arg5) -> (tensor<1xf32>) {
          %15 = affine.min #map(%arg4)
          %16 = affine.min #map(%arg4)
          %extracted_slice = tensor.extract_slice %1#0[%arg4, %arg6] [%15, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
          %extracted_slice_197 = tensor.extract_slice %arg7[%arg4] [%16] [1] : tensor<1xf32> to tensor<?xf32>
          %17 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_197 : tensor<?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %18 = arith.mulf %in, %in : f32
            %19 = arith.addf %out, %18 : f32
            linalg.yield %19 : f32
          } -> tensor<?xf32>
          %inserted_slice = tensor.insert_slice %17 into %arg7[%arg4] [%16] [1] : tensor<?xf32> into tensor<1xf32>
          scf.yield %inserted_slice : tensor<1xf32>
        }
        scf.yield %14 : tensor<1xf32>
      }
      %5 = tensor.empty() : tensor<1x32000xf32>
      %c0_171 = arith.constant 0 : index
      %c1_172 = arith.constant 1 : index
      %c8_173 = arith.constant 8 : index
      %c0_174 = arith.constant 0 : index
      %c32000 = arith.constant 32000 : index
      %c8_175 = arith.constant 8 : index
      %6 = scf.for %arg4 = %c0_171 to %c1_172 step %c8_173 iter_args(%arg5 = %5) -> (tensor<1x32000xf32>) {
        %14 = scf.for %arg6 = %c0_174 to %c32000 step %c8_175 iter_args(%arg7 = %arg5) -> (tensor<1x32000xf32>) {
          %15 = affine.min #map(%arg4)
          %extracted_slice = tensor.extract_slice %arg7[%arg4, %arg6] [%15, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
          %16 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_160 : f32
          } -> tensor<?x8xf32>
          %inserted_slice = tensor.insert_slice %16 into %arg7[%arg4, %arg6] [%15, 8] [1, 1] : tensor<?x8xf32> into tensor<1x32000xf32>
          scf.yield %inserted_slice : tensor<1x32000xf32>
        }
        scf.yield %14 : tensor<1x32000xf32>
      }
      %c0_176 = arith.constant 0 : index
      %c1_177 = arith.constant 1 : index
      %c8_178 = arith.constant 8 : index
      %c0_179 = arith.constant 0 : index
      %c32000_180 = arith.constant 32000 : index
      %c8_181 = arith.constant 8 : index
      %c0_182 = arith.constant 0 : index
      %c768_183 = arith.constant 768 : index
      %c8_184 = arith.constant 8 : index
      %7 = scf.for %arg4 = %c0_176 to %c1_177 step %c8_178 iter_args(%arg5 = %6) -> (tensor<1x32000xf32>) {
        %14 = scf.for %arg6 = %c0_179 to %c32000_180 step %c8_181 iter_args(%arg7 = %arg5) -> (tensor<1x32000xf32>) {
          %15 = scf.for %arg8 = %c0_182 to %c768_183 step %c8_184 iter_args(%arg9 = %arg7) -> (tensor<1x32000xf32>) {
            %16 = affine.min #map(%arg4)
            %17 = affine.min #map(%arg4)
            %18 = affine.min #map(%arg4)
            %extracted_slice = tensor.extract_slice %1#0[%arg4, %arg8] [%16, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_197 = tensor.extract_slice %4[%arg4] [%17] [1] : tensor<1xf32> to tensor<?xf32>
            %extracted_slice_198 = tensor.extract_slice %arg9[%arg4, %arg6] [%18, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
            %19 = linalg.generic {indexing_maps = [#map4, #map11, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_197 : tensor<?x8xf32>, tensor<?xf32>) outs(%extracted_slice_198 : tensor<?x8xf32>) {
            ^bb0(%in: f32, %in_199: f32, %out: f32):
              %20 = arith.divf %in_199, %cst_150 : f32
              %21 = arith.addf %20, %cst_158 : f32
              %22 = math.rsqrt %21 : f32
              %23 = arith.mulf %in, %22 : f32
              %24 = arith.mulf %23, %cst_134 : f32
              %25 = arith.mulf %24, %cst : f32
              %26 = arith.addf %out, %25 : f32
              linalg.yield %26 : f32
            } -> tensor<?x8xf32>
            %inserted_slice = tensor.insert_slice %19 into %arg9[%arg4, %arg6] [%18, 8] [1, 1] : tensor<?x8xf32> into tensor<1x32000xf32>
            scf.yield %inserted_slice : tensor<1x32000xf32>
          }
          scf.yield %15 : tensor<1x32000xf32>
        }
        scf.yield %14 : tensor<1x32000xf32>
      }
      %8 = tensor.empty() : tensor<1xf32>
      %c0_185 = arith.constant 0 : index
      %c1_186 = arith.constant 1 : index
      %c8_187 = arith.constant 8 : index
      %9 = scf.for %arg4 = %c0_185 to %c1_186 step %c8_187 iter_args(%arg5 = %8) -> (tensor<1xf32>) {
        %14 = affine.min #map(%arg4)
        %extracted_slice = tensor.extract_slice %arg5[%arg4] [%14] [1] : tensor<1xf32> to tensor<?xf32>
        %15 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_140 : f32
        } -> tensor<?xf32>
        %inserted_slice = tensor.insert_slice %15 into %arg5[%arg4] [%14] [1] : tensor<?xf32> into tensor<1xf32>
        scf.yield %inserted_slice : tensor<1xf32>
      }
      %10 = tensor.empty() : tensor<1xi64>
      %c0_188 = arith.constant 0 : index
      %c1_189 = arith.constant 1 : index
      %c8_190 = arith.constant 8 : index
      %11 = scf.for %arg4 = %c0_188 to %c1_189 step %c8_190 iter_args(%arg5 = %10) -> (tensor<1xi64>) {
        %14 = affine.min #map(%arg4)
        %extracted_slice = tensor.extract_slice %arg5[%arg4] [%14] [1] : tensor<1xi64> to tensor<?xi64>
        %15 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xi64>) {
        ^bb0(%out: i64):
          linalg.yield %c0_i64 : i64
        } -> tensor<?xi64>
        %inserted_slice = tensor.insert_slice %15 into %arg5[%arg4] [%14] [1] : tensor<?xi64> into tensor<1xi64>
        scf.yield %inserted_slice : tensor<1xi64>
      }
      %c0_191 = arith.constant 0 : index
      %c1_192 = arith.constant 1 : index
      %c8_193 = arith.constant 8 : index
      %c0_194 = arith.constant 0 : index
      %c32000_195 = arith.constant 32000 : index
      %c8_196 = arith.constant 8 : index
      %12:2 = scf.for %arg4 = %c0_191 to %c1_192 step %c8_193 iter_args(%arg5 = %9, %arg6 = %11) -> (tensor<1xf32>, tensor<1xi64>) {
        %14:2 = scf.for %arg7 = %c0_194 to %c32000_195 step %c8_196 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<1xf32>, tensor<1xi64>) {
          %15 = affine.min #map(%arg4)
          %16 = affine.min #map(%arg4)
          %17 = affine.min #map(%arg4)
          %extracted_slice = tensor.extract_slice %7[%arg4, %arg7] [%15, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
          %extracted_slice_197 = tensor.extract_slice %arg8[%arg4] [%16] [1] : tensor<1xf32> to tensor<?xf32>
          %extracted_slice_198 = tensor.extract_slice %arg9[%arg4] [%17] [1] : tensor<1xi64> to tensor<?xi64>
          %18:2 = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_197, %extracted_slice_198 : tensor<?xf32>, tensor<?xi64>) {
          ^bb0(%in: f32, %out: f32, %out_200: i64):
            %19 = linalg.index 1 : index
            %20 = affine.apply #map6(%19, %arg7)
            %21 = arith.index_cast %20 : index to i64
            %22 = arith.cmpf ogt, %in, %out : f32
            %23 = arith.select %22, %in, %out : f32
            %24 = arith.select %22, %21, %out_200 : i64
            linalg.yield %23, %24 : f32, i64
          } -> (tensor<?xf32>, tensor<?xi64>)
          %inserted_slice = tensor.insert_slice %18#0 into %arg8[%arg4] [%16] [1] : tensor<?xf32> into tensor<1xf32>
          %inserted_slice_199 = tensor.insert_slice %18#1 into %arg9[%arg4] [%17] [1] : tensor<?xi64> into tensor<1xi64>
          scf.yield %inserted_slice, %inserted_slice_199 : tensor<1xf32>, tensor<1xi64>
        }
        scf.yield %14#0, %14#1 : tensor<1xf32>, tensor<1xi64>
      }
      %cast = tensor.cast %7 : tensor<1x32000xf32> to tensor<*xf32>
      func.call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
      %extracted = tensor.extract %12#1[%c0] : tensor<1xi64>
      %13 = arith.addi %arg1, %c1_i64 : i64
      scf.yield %extracted, %13, %1#1, %1#2 : i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
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
#map6 = affine_map<(d0) -> (-d0 + 12, 8)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map9 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map10 = affine_map<(d0, d1, d2) -> (d0)>
#map11 = affine_map<(d0, d1, d2) -> (d2, d1)>
module {
  memref.global "private" constant @__constant_12x1024x768xf32 : memref<12x1024x768xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x12x64xf32 : memref<1x12x64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x768xf32 : memref<1x768xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64_1 : memref<3xi64> = dense<[1, 12, 64]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_0 : memref<2xi64> = dense<[1, 768]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64_0 : memref<3xi64> = dense<[1, 1, 768]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[1024, 768]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64 : memref<3xi64> = dense<[1, 1, 64]> {alignment = 64 : i64}
  func.func private @printMemrefF32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c10_i64 = arith.constant 10 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c12 = arith.constant 12 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c64_i64 = arith.constant 64 : i64
    %cst_1 = arith.constant 1.250000e-01 : f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %cst_3 = arith.constant 1.000000e+04 : f32
    %cst_4 = arith.constant 6.400000e+01 : f32
    %cst_5 = arith.constant -2.000000e+00 : f32
    %cst_6 = arith.constant -1.000000e+09 : f32
    %cst_7 = arith.constant 5.000000e-01 : f32
    %cst_8 = arith.constant 7.680000e+02 : f32
    %c32000 = arith.constant 32000 : index
    %c2048 = arith.constant 2048 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c768 = arith.constant 768 : index
    %c8 = arith.constant 8 : index
    %cst_9 = arith.constant 1.300000e+01 : f32
    %cst_10 = arith.constant 1.200000e+01 : f32
    %cst_11 = arith.constant 1.000000e+01 : f32
    %cst_12 = arith.constant 1.100000e+01 : f32
    %cst_13 = arith.constant 9.000000e+00 : f32
    %cst_14 = arith.constant 8.000000e+00 : f32
    %cst_15 = arith.constant 7.000000e+00 : f32
    %cst_16 = arith.constant 0xFF800000 : f32
    %cst_17 = arith.constant 6.000000e+00 : f32
    %cst_18 = arith.constant 5.000000e+00 : f32
    %cst_19 = arith.constant 4.000000e+00 : f32
    %cst_20 = arith.constant 3.000000e+00 : f32
    %0 = memref.get_global @__constant_3xi64 : memref<3xi64>
    %1 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %2 = memref.get_global @__constant_3xi64_0 : memref<3xi64>
    %3 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %4 = memref.get_global @__constant_3xi64_1 : memref<3xi64>
    %5 = memref.get_global @__constant_1x768xf32 : memref<1x768xf32>
    %6 = memref.get_global @__constant_1x12x64xf32 : memref<1x12x64xf32>
    %7 = memref.get_global @__constant_12x1024x768xf32 : memref<12x1024x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %7, %alloc : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %7, %alloc_21 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %8:3 = scf.while (%arg0 = %c0_i64, %arg1 = %alloc, %arg2 = %alloc_21) : (i64, memref<12x1024x768xf32>, memref<12x1024x768xf32>) -> (i64, memref<12x1024x768xf32>, memref<12x1024x768xf32>) {
      %9 = arith.cmpi slt, %arg0, %c10_i64 : i64
      %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
      memref.copy %arg1, %alloc_22 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
      %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
      memref.copy %arg2, %alloc_23 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
      scf.condition(%9) %arg0, %alloc_22, %alloc_23 : i64, memref<12x1024x768xf32>, memref<12x1024x768xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: memref<12x1024x768xf32>, %arg2: memref<12x1024x768xf32>):
      %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
      memref.copy %5, %alloc_22 : memref<1x768xf32> to memref<1x768xf32>
      %9:3 = scf.for %arg3 = %c0 to %c12 step %c1 iter_args(%arg4 = %alloc_22, %arg5 = %arg1, %arg6 = %arg2) -> (memref<1x768xf32>, memref<12x1024x768xf32>, memref<12x1024x768xf32>) {
        %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_27 : memref<1xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        }
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %arg4[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%alloc_27 : memref<1xf32>) {
          ^bb0(%in: f32, %out: f32):
            %13 = arith.mulf %in, %in : f32
            %14 = arith.addf %out, %13 : f32
            linalg.yield %14 : f32
          }
        }
        %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %arg4[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %alloc_28[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_101, %alloc_27 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<1xf32>) outs(%subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_103: f32, %out: f32):
            %13 = arith.divf %in_103, %cst_8 : f32
            %14 = arith.addf %13, %cst_2 : f32
            %15 = math.rsqrt %14 : f32
            %16 = arith.mulf %in, %15 : f32
            %17 = arith.mulf %16, %cst_20 : f32
            linalg.yield %17 : f32
          }
          memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_29[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_29, %alloc_30 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_28[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_30[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %13 = arith.mulf %in, %cst_19 : f32
              %14 = arith.addf %out, %13 : f32
              linalg.yield %14 : f32
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_31[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_31, %alloc_32 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_28[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_32[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %13 = arith.mulf %in, %cst_18 : f32
              %14 = arith.addf %out, %13 : f32
              linalg.yield %14 : f32
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_33[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_33, %alloc_34 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_28[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_34[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %13 = arith.mulf %in, %cst_17 : f32
              %14 = arith.addf %out, %13 : f32
              linalg.yield %14 : f32
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %reshape = memref.reshape %alloc_30(%4) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x12x64xf32>
        %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %11 = arith.uitofp %arg0 : i64 to f32
        scf.for %arg7 = %c0 to %c32 step %c8 {
          %subview_101 = memref.subview %alloc_35[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          %subview_102 = memref.subview %alloc_36[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} outs(%subview_101, %subview_102 : memref<8xf32, strided<[1], offset: ?>>, memref<8xf32, strided<[1], offset: ?>>) {
          ^bb0(%out: f32, %out_103: f32):
            %13 = linalg.index 0 : index
            %14 = affine.apply #map5(%13, %arg7)
            %15 = arith.index_cast %14 : index to i64
            %16 = arith.uitofp %15 : i64 to f32
            %17 = arith.mulf %16, %cst_5 : f32
            %18 = arith.divf %17, %cst_4 : f32
            %19 = math.powf %cst_3, %18 : f32
            %20 = arith.mulf %11, %19 : f32
            %21 = math.cos %20 : f32
            %22 = math.sin %20 : f32
            linalg.yield %21, %22 : f32, f32
          }
          memref.copy %subview_101, %subview_101 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
          memref.copy %subview_102, %subview_102 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        }
        %expand_shape = memref.expand_shape %reshape [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : memref<1x12x64xf32> into memref<1x12x32x2xf32>
        %subview = memref.subview %expand_shape[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %subview_37 = memref.subview %expand_shape[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %expand_shape_38 = memref.expand_shape %alloc_35 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %expand_shape_39 = memref.expand_shape %alloc_36 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        scf.for %arg7 = %c0 to %c12 step %c8 {
          scf.for %arg8 = %c0 to %c32 step %c8 {
            %13 = affine.min #map6(%arg7)
            %subview_101 = memref.subview %subview[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_102 = memref.subview %subview_37[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_103 = memref.subview %expand_shape_38[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_104 = memref.subview %expand_shape_39[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_105 = memref.subview %alloc_40[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            %subview_106 = memref.subview %alloc_41[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map7, #map7, #map8, #map8, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview_101, %subview_102, %subview_103, %subview_104 : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>, memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>, memref<8x1xf32, strided<[1, 1], offset: ?>>, memref<8x1xf32, strided<[1, 1], offset: ?>>) outs(%subview_105, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>, memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_107: f32, %in_108: f32, %in_109: f32, %out: f32, %out_110: f32):
              %14 = arith.mulf %in, %in_108 : f32
              %15 = arith.mulf %in_107, %in_109 : f32
              %16 = arith.subf %14, %15 : f32
              %17 = arith.mulf %in_107, %in_108 : f32
              %18 = arith.mulf %in, %in_109 : f32
              %19 = arith.addf %17, %18 : f32
              linalg.yield %16, %19 : f32, f32
            }
            memref.copy %subview_105, %subview_105 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            memref.copy %subview_106, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
          }
        }
        %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        %subview_43 = memref.subview %alloc_42[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        memref.copy %alloc_40, %subview_43 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        memref.copy %alloc_42, %alloc_44 : memref<1x12x32x2xf32> to memref<1x12x32x2xf32>
        %subview_45 = memref.subview %alloc_44[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        memref.copy %alloc_41, %subview_45 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape = memref.collapse_shape %alloc_44 [[0], [1], [2, 3]] : memref<1x12x32x2xf32> into memref<1x12x64xf32>
        %reshape_46 = memref.reshape %collapse_shape(%3) : (memref<1x12x64xf32>, memref<2xi64>) -> memref<1x768xf32>
        %reshape_47 = memref.reshape %alloc_32(%4) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x12x64xf32>
        %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        scf.for %arg7 = %c0 to %c32 step %c8 {
          %subview_101 = memref.subview %alloc_48[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          %subview_102 = memref.subview %alloc_49[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} outs(%subview_101, %subview_102 : memref<8xf32, strided<[1], offset: ?>>, memref<8xf32, strided<[1], offset: ?>>) {
          ^bb0(%out: f32, %out_103: f32):
            %13 = linalg.index 0 : index
            %14 = affine.apply #map5(%13, %arg7)
            %15 = arith.index_cast %14 : index to i64
            %16 = arith.uitofp %15 : i64 to f32
            %17 = arith.mulf %16, %cst_5 : f32
            %18 = arith.divf %17, %cst_4 : f32
            %19 = math.powf %cst_3, %18 : f32
            %20 = arith.mulf %11, %19 : f32
            %21 = math.cos %20 : f32
            %22 = math.sin %20 : f32
            linalg.yield %21, %22 : f32, f32
          }
          memref.copy %subview_101, %subview_101 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
          memref.copy %subview_102, %subview_102 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        }
        %expand_shape_50 = memref.expand_shape %reshape_47 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : memref<1x12x64xf32> into memref<1x12x32x2xf32>
        %subview_51 = memref.subview %expand_shape_50[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %subview_52 = memref.subview %expand_shape_50[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %expand_shape_53 = memref.expand_shape %alloc_48 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %expand_shape_54 = memref.expand_shape %alloc_49 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        scf.for %arg7 = %c0 to %c12 step %c8 {
          scf.for %arg8 = %c0 to %c32 step %c8 {
            %13 = affine.min #map6(%arg7)
            %subview_101 = memref.subview %subview_51[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_102 = memref.subview %subview_52[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_103 = memref.subview %expand_shape_53[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_104 = memref.subview %expand_shape_54[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_105 = memref.subview %alloc_55[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            %subview_106 = memref.subview %alloc_56[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map7, #map7, #map8, #map8, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview_101, %subview_102, %subview_103, %subview_104 : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>, memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>, memref<8x1xf32, strided<[1, 1], offset: ?>>, memref<8x1xf32, strided<[1, 1], offset: ?>>) outs(%subview_105, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>, memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_107: f32, %in_108: f32, %in_109: f32, %out: f32, %out_110: f32):
              %14 = arith.mulf %in, %in_108 : f32
              %15 = arith.mulf %in_107, %in_109 : f32
              %16 = arith.subf %14, %15 : f32
              %17 = arith.mulf %in_107, %in_108 : f32
              %18 = arith.mulf %in, %in_109 : f32
              %19 = arith.addf %17, %18 : f32
              linalg.yield %16, %19 : f32, f32
            }
            memref.copy %subview_105, %subview_105 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            memref.copy %subview_106, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
          }
        }
        %alloc_57 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        %subview_58 = memref.subview %alloc_57[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        memref.copy %alloc_55, %subview_58 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        memref.copy %alloc_57, %alloc_59 : memref<1x12x32x2xf32> to memref<1x12x32x2xf32>
        %subview_60 = memref.subview %alloc_59[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        memref.copy %alloc_56, %subview_60 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape_61 = memref.collapse_shape %alloc_59 [[0], [1], [2, 3]] : memref<1x12x32x2xf32> into memref<1x12x64xf32>
        %reshape_62 = memref.reshape %alloc_34(%4) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x12x64xf32>
        %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %alloc_64 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        scf.for %arg7 = %c0 to %c32 step %c8 {
          %subview_101 = memref.subview %alloc_63[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          %subview_102 = memref.subview %alloc_64[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} outs(%subview_101, %subview_102 : memref<8xf32, strided<[1], offset: ?>>, memref<8xf32, strided<[1], offset: ?>>) {
          ^bb0(%out: f32, %out_103: f32):
            %13 = linalg.index 0 : index
            %14 = affine.apply #map5(%13, %arg7)
            %15 = arith.index_cast %14 : index to i64
            %16 = arith.uitofp %15 : i64 to f32
            %17 = arith.mulf %16, %cst_5 : f32
            %18 = arith.divf %17, %cst_4 : f32
            %19 = math.powf %cst_3, %18 : f32
            %20 = arith.mulf %11, %19 : f32
            %21 = math.cos %20 : f32
            %22 = math.sin %20 : f32
            linalg.yield %21, %22 : f32, f32
          }
          memref.copy %subview_101, %subview_101 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
          memref.copy %subview_102, %subview_102 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        }
        %expand_shape_65 = memref.expand_shape %reshape_62 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : memref<1x12x64xf32> into memref<1x12x32x2xf32>
        %subview_66 = memref.subview %expand_shape_65[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %subview_67 = memref.subview %expand_shape_65[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %expand_shape_68 = memref.expand_shape %alloc_63 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %expand_shape_69 = memref.expand_shape %alloc_64 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        scf.for %arg7 = %c0 to %c12 step %c8 {
          scf.for %arg8 = %c0 to %c32 step %c8 {
            %13 = affine.min #map6(%arg7)
            %subview_101 = memref.subview %subview_66[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_102 = memref.subview %subview_67[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_103 = memref.subview %expand_shape_68[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_104 = memref.subview %expand_shape_69[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_105 = memref.subview %alloc_70[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            %subview_106 = memref.subview %alloc_71[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map7, #map7, #map8, #map8, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview_101, %subview_102, %subview_103, %subview_104 : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>, memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>, memref<8x1xf32, strided<[1, 1], offset: ?>>, memref<8x1xf32, strided<[1, 1], offset: ?>>) outs(%subview_105, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>, memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_107: f32, %in_108: f32, %in_109: f32, %out: f32, %out_110: f32):
              %14 = arith.mulf %in, %in_108 : f32
              %15 = arith.mulf %in_107, %in_109 : f32
              %16 = arith.subf %14, %15 : f32
              %17 = arith.mulf %in_107, %in_108 : f32
              %18 = arith.mulf %in, %in_109 : f32
              %19 = arith.addf %17, %18 : f32
              linalg.yield %16, %19 : f32, f32
            }
            memref.copy %subview_105, %subview_105 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            memref.copy %subview_106, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
          }
        }
        %alloc_72 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        %subview_73 = memref.subview %alloc_72[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        memref.copy %alloc_70, %subview_73 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %alloc_74 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        memref.copy %alloc_72, %alloc_74 : memref<1x12x32x2xf32> to memref<1x12x32x2xf32>
        %subview_75 = memref.subview %alloc_74[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        memref.copy %alloc_71, %subview_75 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape_76 = memref.collapse_shape %alloc_74 [[0], [1], [2, 3]] : memref<1x12x32x2xf32> into memref<1x12x64xf32>
        %reshape_77 = memref.reshape %collapse_shape_61(%2) : (memref<1x12x64xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %12 = arith.index_cast %arg0 : i64 to index
        %alloc_78 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
        memref.copy %arg5, %alloc_78 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
        %subview_79 = memref.subview %alloc_78[%arg3, %12, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape_77, %subview_79 : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        %reshape_80 = memref.reshape %collapse_shape_76(%2) : (memref<1x12x64xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %alloc_81 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
        memref.copy %arg6, %alloc_81 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
        %subview_82 = memref.subview %alloc_81[%arg3, %12, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape_80, %subview_82 : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_83 = memref.subview %alloc_78[%arg3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1024x768xf32, strided<[786432, 768, 1], offset: ?>>
        %alloc_84 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x768xf32>
        memref.copy %subview_83, %alloc_84 : memref<1x1024x768xf32, strided<[786432, 768, 1], offset: ?>> to memref<1x1024x768xf32>
        %reshape_85 = memref.reshape %alloc_84(%1) : (memref<1x1024x768xf32>, memref<2xi64>) -> memref<1024x768xf32>
        %subview_86 = memref.subview %alloc_81[%arg3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1024x768xf32, strided<[786432, 768, 1], offset: ?>>
        %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x768xf32>
        memref.copy %subview_86, %alloc_87 : memref<1x1024x768xf32, strided<[786432, 768, 1], offset: ?>> to memref<1x1024x768xf32>
        %reshape_88 = memref.reshape %alloc_87(%1) : (memref<1x1024x768xf32>, memref<2xi64>) -> memref<1024x768xf32>
        %alloc_89 = memref.alloc() {alignment = 64 : i64} : memref<1x12x64xf32>
        memref.copy %6, %alloc_89 : memref<1x12x64xf32> to memref<1x12x64xf32>
        scf.for %arg7 = %c0 to %c12 step %c1 {
          %13 = arith.index_cast %arg7 : index to i64
          %14 = arith.muli %13, %c64_i64 : i64
          %15 = arith.index_cast %14 : i64 to index
          %subview_101 = memref.subview %reshape_46[0, %15] [1, 64] [1, 1] : memref<1x768xf32> to memref<1x64xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %reshape_85[0, %15] [1024, 64] [1, 1] : memref<1024x768xf32> to memref<1024x64xf32, strided<[768, 1], offset: ?>>
          %alloc_103 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_103[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_113 : memref<1x8xf32, strided<[1024, 1], offset: ?>>) {
            ^bb0(%out: f32):
              linalg.yield %cst_0 : f32
            }
            memref.copy %subview_113, %subview_113 : memref<1x8xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
          }
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            scf.for %arg9 = %c0 to %c64 step %c8 {
              %subview_113 = memref.subview %subview_101[0, %arg9] [1, 8] [1, 1] : memref<1x64xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
              %subview_114 = memref.subview %subview_102[%arg8, %arg9] [8, 8] [1, 1] : memref<1024x64xf32, strided<[768, 1], offset: ?>> to memref<8x8xf32, strided<[768, 1], offset: ?>>
              %subview_115 = memref.subview %alloc_103[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
              linalg.generic {indexing_maps = [#map3, #map9, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_113, %subview_114 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<8x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_115 : memref<1x8xf32, strided<[1024, 1], offset: ?>>) {
              ^bb0(%in: f32, %in_116: f32, %out: f32):
                %16 = linalg.index 1 : index
                %17 = affine.apply #map5(%16, %arg8)
                %18 = arith.index_cast %17 : index to i64
                %19 = arith.cmpi sle, %18, %arg0 : i64
                %20 = arith.select %19, %cst, %cst_0 : f32
                %21 = arith.mulf %in, %in_116 : f32
                %22 = arith.addf %out, %21 : f32
                %23 = arith.cmpf ugt, %20, %cst_7 : f32
                %24 = arith.select %23, %22, %cst_6 : f32
                linalg.yield %24 : f32
              }
              memref.copy %subview_115, %subview_115 : memref<1x8xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            }
          }
          %alloc_104 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_103[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            %subview_114 = memref.subview %alloc_104[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_113 : memref<1x8xf32, strided<[1024, 1], offset: ?>>) outs(%subview_114 : memref<1x8xf32, strided<[1024, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %16 = arith.mulf %in, %cst_1 : f32
              linalg.yield %16 : f32
            }
            memref.copy %subview_114, %subview_114 : memref<1x8xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
          }
          %alloc_105 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_105 : memref<1xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_16 : f32
          }
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_104[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview_113 : memref<1x8xf32, strided<[1024, 1], offset: ?>>) outs(%alloc_105 : memref<1xf32>) {
            ^bb0(%in: f32, %out: f32):
              %16 = arith.maxnumf %in, %out : f32
              linalg.yield %16 : f32
            }
          }
          %alloc_106 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_104[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            %subview_114 = memref.subview %alloc_106[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_113, %alloc_105 : memref<1x8xf32, strided<[1024, 1], offset: ?>>, memref<1xf32>) outs(%subview_114 : memref<1x8xf32, strided<[1024, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_115: f32, %out: f32):
              %16 = arith.subf %in, %in_115 : f32
              %17 = math.exp %16 : f32
              linalg.yield %17 : f32
            }
            memref.copy %subview_114, %subview_114 : memref<1x8xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
          }
          %alloc_107 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_107 : memref<1xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          }
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_106[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_113 : memref<1x8xf32, strided<[1024, 1], offset: ?>>) outs(%alloc_107 : memref<1xf32>) {
            ^bb0(%in: f32, %out: f32):
              %16 = arith.addf %in, %out : f32
              linalg.yield %16 : f32
            }
          }
          %subview_108 = memref.subview %reshape_88[0, %15] [1024, 64] [1, 1] : memref<1024x768xf32> to memref<1024x64xf32, strided<[768, 1], offset: ?>>
          %alloc_109 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
          scf.for %arg8 = %c0 to %c64 step %c8 {
            %subview_113 = memref.subview %alloc_109[0, %arg8] [1, 8] [1, 1] : memref<1x64xf32> to memref<1x8xf32, strided<[64, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_113 : memref<1x8xf32, strided<[64, 1], offset: ?>>) {
            ^bb0(%out: f32):
              linalg.yield %cst_0 : f32
            }
            memref.copy %subview_113, %subview_113 : memref<1x8xf32, strided<[64, 1], offset: ?>> to memref<1x8xf32, strided<[64, 1], offset: ?>>
          }
          %alloc_110 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
          memref.copy %alloc_109, %alloc_110 : memref<1x64xf32> to memref<1x64xf32>
          scf.for %arg8 = %c0 to %c64 step %c8 {
            scf.for %arg9 = %c0 to %c1024 step %c8 {
              %subview_113 = memref.subview %alloc_106[0, %arg9] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
              %subview_114 = memref.subview %subview_108[%arg9, %arg8] [8, 8] [1, 1] : memref<1024x64xf32, strided<[768, 1], offset: ?>> to memref<8x8xf32, strided<[768, 1], offset: ?>>
              %subview_115 = memref.subview %alloc_110[0, %arg8] [1, 8] [1, 1] : memref<1x64xf32> to memref<1x8xf32, strided<[64, 1], offset: ?>>
              linalg.generic {indexing_maps = [#map3, #map10, #map11, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_113, %alloc_107, %subview_114 : memref<1x8xf32, strided<[1024, 1], offset: ?>>, memref<1xf32>, memref<8x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_115 : memref<1x8xf32, strided<[64, 1], offset: ?>>) {
              ^bb0(%in: f32, %in_116: f32, %in_117: f32, %out: f32):
                %16 = arith.divf %in, %in_116 : f32
                %17 = arith.mulf %16, %in_117 : f32
                %18 = arith.addf %out, %17 : f32
                linalg.yield %18 : f32
              }
              memref.copy %subview_115, %subview_115 : memref<1x8xf32, strided<[64, 1], offset: ?>> to memref<1x8xf32, strided<[64, 1], offset: ?>>
            }
          }
          %reshape_111 = memref.reshape %alloc_110(%0) : (memref<1x64xf32>, memref<3xi64>) -> memref<1x1x64xf32>
          %subview_112 = memref.subview %alloc_89[0, %arg7, 0] [1, 1, 64] [1, 1, 1] : memref<1x12x64xf32> to memref<1x1x64xf32, strided<[768, 64, 1], offset: ?>>
          memref.copy %reshape_111, %subview_112 : memref<1x1x64xf32> to memref<1x1x64xf32, strided<[768, 64, 1], offset: ?>>
        }
        %reshape_90 = memref.reshape %alloc_89(%3) : (memref<1x12x64xf32>, memref<2xi64>) -> memref<1x768xf32>
        %alloc_91 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_91[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %reshape_90[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_91[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %13 = arith.mulf %in, %cst_15 : f32
              %14 = arith.addf %out, %13 : f32
              linalg.yield %14 : f32
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_92 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %arg4[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %alloc_91[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_103 = memref.subview %alloc_92[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_101, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_104: f32, %out: f32):
            %13 = arith.addf %in, %in_104 : f32
            linalg.yield %13 : f32
          }
          memref.copy %subview_103, %subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_93 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_93 : memref<1xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        }
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_92[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%alloc_93 : memref<1xf32>) {
          ^bb0(%in: f32, %out: f32):
            %13 = arith.mulf %in, %in : f32
            %14 = arith.addf %out, %13 : f32
            linalg.yield %14 : f32
          }
        }
        %alloc_94 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_92[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %alloc_94[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_101, %alloc_93 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<1xf32>) outs(%subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_103: f32, %out: f32):
            %13 = arith.divf %in_103, %cst_8 : f32
            %14 = arith.addf %13, %cst_2 : f32
            %15 = math.rsqrt %14 : f32
            %16 = arith.mulf %in, %15 : f32
            %17 = arith.mulf %16, %cst_14 : f32
            linalg.yield %17 : f32
          }
          memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<1x2048xf32>
        scf.for %arg7 = %c0 to %c2048 step %c8 {
          %subview_101 = memref.subview %alloc_95[0, %arg7] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_101 : memref<1x8xf32, strided<[2048, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[2048, 1], offset: ?>> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
        }
        scf.for %arg7 = %c0 to %c2048 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_94[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_95[0, %arg7] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_102 : memref<1x8xf32, strided<[2048, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %13 = arith.mulf %in, %cst_13 : f32
              %14 = arith.addf %out, %13 : f32
              linalg.yield %14 : f32
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[2048, 1], offset: ?>> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
          }
        }
        %alloc_96 = memref.alloc() {alignment = 64 : i64} : memref<1x2048xf32>
        scf.for %arg7 = %c0 to %c2048 step %c8 {
          %subview_101 = memref.subview %alloc_96[0, %arg7] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_101 : memref<1x8xf32, strided<[2048, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[2048, 1], offset: ?>> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
        }
        scf.for %arg7 = %c0 to %c2048 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_94[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_96[0, %arg7] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_102 : memref<1x8xf32, strided<[2048, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %13 = arith.mulf %in, %cst_12 : f32
              %14 = arith.addf %out, %13 : f32
              linalg.yield %14 : f32
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[2048, 1], offset: ?>> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
          }
        }
        %alloc_97 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_97[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c2048 step %c8 {
            %subview_101 = memref.subview %alloc_95[0, %arg8] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_96[0, %arg8] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
            %subview_103 = memref.subview %alloc_97[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_101, %subview_102 : memref<1x8xf32, strided<[2048, 1], offset: ?>>, memref<1x8xf32, strided<[2048, 1], offset: ?>>) outs(%subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_104: f32, %out: f32):
              %13 = arith.negf %in : f32
              %14 = math.exp %13 : f32
              %15 = arith.addf %in, %14 : f32
              %16 = arith.divf %in, %15 : f32
              %17 = arith.mulf %16, %in_104 : f32
              %18 = arith.mulf %17, %cst_11 : f32
              %19 = arith.addf %out, %18 : f32
              linalg.yield %19 : f32
            }
            memref.copy %subview_103, %subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_98 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_92[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %alloc_97[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_103 = memref.subview %alloc_98[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_101, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_104: f32, %out: f32):
            %13 = arith.addf %in, %in_104 : f32
            linalg.yield %13 : f32
          }
          memref.copy %subview_103, %subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_99 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
        memref.copy %alloc_78, %alloc_99 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
        %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
        memref.copy %alloc_81, %alloc_100 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
        scf.yield %alloc_98, %alloc_99, %alloc_100 : memref<1x768xf32>, memref<12x1024x768xf32>, memref<12x1024x768xf32>
      }
      %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_23 : memref<1xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg3 = %c0 to %c768 step %c8 {
        %subview = memref.subview %9#0[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%alloc_23 : memref<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.mulf %in, %in : f32
          %12 = arith.addf %out, %11 : f32
          linalg.yield %12 : f32
        }
      }
      %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x32000xf32>
      scf.for %arg3 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_24[0, %arg3] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<1x8xf32, strided<[32000, 1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        }
        memref.copy %subview, %subview : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
      }
      scf.for %arg3 = %c0 to %c32000 step %c8 {
        scf.for %arg4 = %c0 to %c768 step %c8 {
          %subview = memref.subview %9#0[0, %arg4] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_27 = memref.subview %alloc_24[0, %arg3] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map3, #map10, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview, %alloc_23 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<1xf32>) outs(%subview_27 : memref<1x8xf32, strided<[32000, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_28: f32, %out: f32):
            %11 = arith.divf %in_28, %cst_8 : f32
            %12 = arith.addf %11, %cst_2 : f32
            %13 = math.rsqrt %12 : f32
            %14 = arith.mulf %in, %13 : f32
            %15 = arith.mulf %14, %cst_10 : f32
            %16 = arith.mulf %15, %cst_9 : f32
            %17 = arith.addf %out, %16 : f32
            linalg.yield %17 : f32
          }
          memref.copy %subview_27, %subview_27 : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        }
      }
      %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_25 : memref<1xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_16 : f32
      }
      %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_26 : memref<1xi64>) {
      ^bb0(%out: i64):
        linalg.yield %c0_i64 : i64
      }
      scf.for %arg3 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_24[0, %arg3] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<1x8xf32, strided<[32000, 1], offset: ?>>) outs(%alloc_25, %alloc_26 : memref<1xf32>, memref<1xi64>) {
        ^bb0(%in: f32, %out: f32, %out_27: i64):
          %11 = linalg.index 1 : index
          %12 = affine.apply #map5(%11, %arg3)
          %13 = arith.index_cast %12 : index to i64
          %14 = arith.cmpf ogt, %in, %out : f32
          %15 = arith.select %14, %in, %out : f32
          %16 = arith.select %14, %13, %out_27 : i64
          linalg.yield %15, %16 : f32, i64
        }
      }
      %cast = memref.cast %alloc_24 : memref<1x32000xf32> to memref<*xf32>
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
      %10 = arith.addi %arg0, %c1_i64 : i64
      scf.yield %10, %9#1, %9#2 : i64, memref<12x1024x768xf32>, memref<12x1024x768xf32>
    }
    return
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
#map = affine_map<(d0, d1) -> (d0 + d1)>
#map1 = affine_map<(d0) -> (-d0 + 12, 8)>
module {
  memref.global "private" constant @__constant_12x1024x768xf32 : memref<12x1024x768xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x12x64xf32 : memref<1x12x64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x768xf32 : memref<1x768xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64_1 : memref<3xi64> = dense<[1, 12, 64]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_0 : memref<2xi64> = dense<[1, 768]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64_0 : memref<3xi64> = dense<[1, 1, 768]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[1024, 768]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64 : memref<3xi64> = dense<[1, 1, 64]> {alignment = 64 : i64}
  func.func private @printMemrefF32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c10_i64 = arith.constant 10 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c12 = arith.constant 12 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c64_i64 = arith.constant 64 : i64
    %cst_1 = arith.constant 1.250000e-01 : f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %cst_3 = arith.constant 1.000000e+04 : f32
    %cst_4 = arith.constant 6.400000e+01 : f32
    %cst_5 = arith.constant -2.000000e+00 : f32
    %cst_6 = arith.constant -1.000000e+09 : f32
    %cst_7 = arith.constant 5.000000e-01 : f32
    %cst_8 = arith.constant 7.680000e+02 : f32
    %c32000 = arith.constant 32000 : index
    %c2048 = arith.constant 2048 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c768 = arith.constant 768 : index
    %c8 = arith.constant 8 : index
    %cst_9 = arith.constant 1.300000e+01 : f32
    %cst_10 = arith.constant 1.200000e+01 : f32
    %cst_11 = arith.constant 1.000000e+01 : f32
    %cst_12 = arith.constant 1.100000e+01 : f32
    %cst_13 = arith.constant 9.000000e+00 : f32
    %cst_14 = arith.constant 8.000000e+00 : f32
    %cst_15 = arith.constant 7.000000e+00 : f32
    %cst_16 = arith.constant 0xFF800000 : f32
    %cst_17 = arith.constant 6.000000e+00 : f32
    %cst_18 = arith.constant 5.000000e+00 : f32
    %cst_19 = arith.constant 4.000000e+00 : f32
    %cst_20 = arith.constant 3.000000e+00 : f32
    %0 = memref.get_global @__constant_3xi64 : memref<3xi64>
    %1 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %2 = memref.get_global @__constant_3xi64_0 : memref<3xi64>
    %3 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %4 = memref.get_global @__constant_3xi64_1 : memref<3xi64>
    %5 = memref.get_global @__constant_1x768xf32 : memref<1x768xf32>
    %6 = memref.get_global @__constant_1x12x64xf32 : memref<1x12x64xf32>
    %7 = memref.get_global @__constant_12x1024x768xf32 : memref<12x1024x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %7, %alloc : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %7, %alloc_21 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %8:3 = scf.while (%arg0 = %c0_i64, %arg1 = %alloc, %arg2 = %alloc_21) : (i64, memref<12x1024x768xf32>, memref<12x1024x768xf32>) -> (i64, memref<12x1024x768xf32>, memref<12x1024x768xf32>) {
      %9 = arith.cmpi slt, %arg0, %c10_i64 : i64
      %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
      memref.copy %arg1, %alloc_22 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
      %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
      memref.copy %arg2, %alloc_23 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
      scf.condition(%9) %arg0, %alloc_22, %alloc_23 : i64, memref<12x1024x768xf32>, memref<12x1024x768xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: memref<12x1024x768xf32>, %arg2: memref<12x1024x768xf32>):
      %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
      memref.copy %5, %alloc_22 : memref<1x768xf32> to memref<1x768xf32>
      %9:3 = scf.for %arg3 = %c0 to %c12 step %c1 iter_args(%arg4 = %alloc_22, %arg5 = %arg1, %arg6 = %arg2) -> (memref<1x768xf32>, memref<12x1024x768xf32>, memref<12x1024x768xf32>) {
        %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        scf.for %arg7 = %c0 to %c1 step %c1 {
          memref.store %cst_0, %alloc_27[%arg7] : memref<1xf32>
        }
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %arg4[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              %13 = memref.load %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %14 = memref.load %alloc_27[%arg8] : memref<1xf32>
              %15 = arith.mulf %13, %13 : f32
              %16 = arith.addf %14, %15 : f32
              memref.store %16, %alloc_27[%arg8] : memref<1xf32>
            }
          }
        }
        %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %arg4[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %alloc_28[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              %13 = memref.load %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %14 = memref.load %alloc_27[%arg8] : memref<1xf32>
              %15 = arith.divf %14, %cst_8 : f32
              %16 = arith.addf %15, %cst_2 : f32
              %17 = math.rsqrt %16 : f32
              %18 = arith.mulf %13, %17 : f32
              %19 = arith.mulf %18, %cst_20 : f32
              memref.store %19, %subview_102[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_29[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              memref.store %cst_0, %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_29, %alloc_30 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_28[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_30[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  %13 = memref.load %subview_101[%arg9, %arg11] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %14 = memref.load %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %15 = arith.mulf %13, %cst_19 : f32
                  %16 = arith.addf %14, %15 : f32
                  memref.store %16, %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_31[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              memref.store %cst_0, %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_31, %alloc_32 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_28[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_32[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  %13 = memref.load %subview_101[%arg9, %arg11] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %14 = memref.load %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %15 = arith.mulf %13, %cst_18 : f32
                  %16 = arith.addf %14, %15 : f32
                  memref.store %16, %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_33[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              memref.store %cst_0, %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_33, %alloc_34 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_28[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_34[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  %13 = memref.load %subview_101[%arg9, %arg11] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %14 = memref.load %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %15 = arith.mulf %13, %cst_17 : f32
                  %16 = arith.addf %14, %15 : f32
                  memref.store %16, %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %reshape = memref.reshape %alloc_30(%4) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x12x64xf32>
        %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %11 = arith.uitofp %arg0 : i64 to f32
        scf.for %arg7 = %c0 to %c32 step %c8 {
          %subview_101 = memref.subview %alloc_35[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          %subview_102 = memref.subview %alloc_36[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %13 = affine.apply #map(%arg8, %arg7)
            %14 = arith.index_cast %13 : index to i64
            %15 = arith.uitofp %14 : i64 to f32
            %16 = arith.mulf %15, %cst_5 : f32
            %17 = arith.divf %16, %cst_4 : f32
            %18 = math.powf %cst_3, %17 : f32
            %19 = arith.mulf %11, %18 : f32
            %20 = math.cos %19 : f32
            %21 = math.sin %19 : f32
            memref.store %20, %subview_101[%arg8] : memref<8xf32, strided<[1], offset: ?>>
            memref.store %21, %subview_102[%arg8] : memref<8xf32, strided<[1], offset: ?>>
          }
          memref.copy %subview_101, %subview_101 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
          memref.copy %subview_102, %subview_102 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        }
        %expand_shape = memref.expand_shape %reshape [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : memref<1x12x64xf32> into memref<1x12x32x2xf32>
        %subview = memref.subview %expand_shape[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %subview_37 = memref.subview %expand_shape[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %expand_shape_38 = memref.expand_shape %alloc_35 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %expand_shape_39 = memref.expand_shape %alloc_36 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        scf.for %arg7 = %c0 to %c12 step %c8 {
          scf.for %arg8 = %c0 to %c32 step %c8 {
            %13 = affine.min #map1(%arg7)
            %subview_101 = memref.subview %subview[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_102 = memref.subview %subview_37[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_103 = memref.subview %expand_shape_38[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_104 = memref.subview %expand_shape_39[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_105 = memref.subview %alloc_40[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            %subview_106 = memref.subview %alloc_41[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %13 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  scf.for %arg12 = %c0 to %c1 step %c1 {
                    %14 = memref.load %subview_101[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
                    %15 = memref.load %subview_102[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
                    %16 = memref.load %subview_103[%arg11, %arg12] : memref<8x1xf32, strided<[1, 1], offset: ?>>
                    %17 = memref.load %subview_104[%arg11, %arg12] : memref<8x1xf32, strided<[1, 1], offset: ?>>
                    %18 = arith.mulf %14, %16 : f32
                    %19 = arith.mulf %15, %17 : f32
                    %20 = arith.subf %18, %19 : f32
                    %21 = arith.mulf %15, %16 : f32
                    %22 = arith.mulf %14, %17 : f32
                    %23 = arith.addf %21, %22 : f32
                    memref.store %20, %subview_105[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
                    memref.store %23, %subview_106[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
                  }
                }
              }
            }
            memref.copy %subview_105, %subview_105 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            memref.copy %subview_106, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
          }
        }
        %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        %subview_43 = memref.subview %alloc_42[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        memref.copy %alloc_40, %subview_43 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        memref.copy %alloc_42, %alloc_44 : memref<1x12x32x2xf32> to memref<1x12x32x2xf32>
        %subview_45 = memref.subview %alloc_44[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        memref.copy %alloc_41, %subview_45 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape = memref.collapse_shape %alloc_44 [[0], [1], [2, 3]] : memref<1x12x32x2xf32> into memref<1x12x64xf32>
        %reshape_46 = memref.reshape %collapse_shape(%3) : (memref<1x12x64xf32>, memref<2xi64>) -> memref<1x768xf32>
        %reshape_47 = memref.reshape %alloc_32(%4) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x12x64xf32>
        %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        scf.for %arg7 = %c0 to %c32 step %c8 {
          %subview_101 = memref.subview %alloc_48[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          %subview_102 = memref.subview %alloc_49[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %13 = affine.apply #map(%arg8, %arg7)
            %14 = arith.index_cast %13 : index to i64
            %15 = arith.uitofp %14 : i64 to f32
            %16 = arith.mulf %15, %cst_5 : f32
            %17 = arith.divf %16, %cst_4 : f32
            %18 = math.powf %cst_3, %17 : f32
            %19 = arith.mulf %11, %18 : f32
            %20 = math.cos %19 : f32
            %21 = math.sin %19 : f32
            memref.store %20, %subview_101[%arg8] : memref<8xf32, strided<[1], offset: ?>>
            memref.store %21, %subview_102[%arg8] : memref<8xf32, strided<[1], offset: ?>>
          }
          memref.copy %subview_101, %subview_101 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
          memref.copy %subview_102, %subview_102 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        }
        %expand_shape_50 = memref.expand_shape %reshape_47 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : memref<1x12x64xf32> into memref<1x12x32x2xf32>
        %subview_51 = memref.subview %expand_shape_50[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %subview_52 = memref.subview %expand_shape_50[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %expand_shape_53 = memref.expand_shape %alloc_48 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %expand_shape_54 = memref.expand_shape %alloc_49 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        scf.for %arg7 = %c0 to %c12 step %c8 {
          scf.for %arg8 = %c0 to %c32 step %c8 {
            %13 = affine.min #map1(%arg7)
            %subview_101 = memref.subview %subview_51[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_102 = memref.subview %subview_52[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_103 = memref.subview %expand_shape_53[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_104 = memref.subview %expand_shape_54[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_105 = memref.subview %alloc_55[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            %subview_106 = memref.subview %alloc_56[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %13 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  scf.for %arg12 = %c0 to %c1 step %c1 {
                    %14 = memref.load %subview_101[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
                    %15 = memref.load %subview_102[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
                    %16 = memref.load %subview_103[%arg11, %arg12] : memref<8x1xf32, strided<[1, 1], offset: ?>>
                    %17 = memref.load %subview_104[%arg11, %arg12] : memref<8x1xf32, strided<[1, 1], offset: ?>>
                    %18 = arith.mulf %14, %16 : f32
                    %19 = arith.mulf %15, %17 : f32
                    %20 = arith.subf %18, %19 : f32
                    %21 = arith.mulf %15, %16 : f32
                    %22 = arith.mulf %14, %17 : f32
                    %23 = arith.addf %21, %22 : f32
                    memref.store %20, %subview_105[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
                    memref.store %23, %subview_106[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
                  }
                }
              }
            }
            memref.copy %subview_105, %subview_105 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            memref.copy %subview_106, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
          }
        }
        %alloc_57 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        %subview_58 = memref.subview %alloc_57[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        memref.copy %alloc_55, %subview_58 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        memref.copy %alloc_57, %alloc_59 : memref<1x12x32x2xf32> to memref<1x12x32x2xf32>
        %subview_60 = memref.subview %alloc_59[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        memref.copy %alloc_56, %subview_60 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape_61 = memref.collapse_shape %alloc_59 [[0], [1], [2, 3]] : memref<1x12x32x2xf32> into memref<1x12x64xf32>
        %reshape_62 = memref.reshape %alloc_34(%4) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x12x64xf32>
        %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %alloc_64 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        scf.for %arg7 = %c0 to %c32 step %c8 {
          %subview_101 = memref.subview %alloc_63[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          %subview_102 = memref.subview %alloc_64[%arg7] [8] [1] : memref<32xf32> to memref<8xf32, strided<[1], offset: ?>>
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %13 = affine.apply #map(%arg8, %arg7)
            %14 = arith.index_cast %13 : index to i64
            %15 = arith.uitofp %14 : i64 to f32
            %16 = arith.mulf %15, %cst_5 : f32
            %17 = arith.divf %16, %cst_4 : f32
            %18 = math.powf %cst_3, %17 : f32
            %19 = arith.mulf %11, %18 : f32
            %20 = math.cos %19 : f32
            %21 = math.sin %19 : f32
            memref.store %20, %subview_101[%arg8] : memref<8xf32, strided<[1], offset: ?>>
            memref.store %21, %subview_102[%arg8] : memref<8xf32, strided<[1], offset: ?>>
          }
          memref.copy %subview_101, %subview_101 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
          memref.copy %subview_102, %subview_102 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        }
        %expand_shape_65 = memref.expand_shape %reshape_62 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : memref<1x12x64xf32> into memref<1x12x32x2xf32>
        %subview_66 = memref.subview %expand_shape_65[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %subview_67 = memref.subview %expand_shape_65[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %expand_shape_68 = memref.expand_shape %alloc_63 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %expand_shape_69 = memref.expand_shape %alloc_64 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
        %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x1xf32>
        scf.for %arg7 = %c0 to %c12 step %c8 {
          scf.for %arg8 = %c0 to %c32 step %c8 {
            %13 = affine.min #map1(%arg7)
            %subview_101 = memref.subview %subview_66[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_102 = memref.subview %subview_67[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>> to memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
            %subview_103 = memref.subview %expand_shape_68[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_104 = memref.subview %expand_shape_69[%arg8, 0] [8, 1] [1, 1] : memref<32x1xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
            %subview_105 = memref.subview %alloc_70[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            %subview_106 = memref.subview %alloc_71[0, %arg7, %arg8, 0] [1, %13, 8, 1] [1, 1, 1, 1] : memref<1x12x32x1xf32> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %13 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  scf.for %arg12 = %c0 to %c1 step %c1 {
                    %14 = memref.load %subview_101[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
                    %15 = memref.load %subview_102[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[768, 64, 2, 1], offset: ?>>
                    %16 = memref.load %subview_103[%arg11, %arg12] : memref<8x1xf32, strided<[1, 1], offset: ?>>
                    %17 = memref.load %subview_104[%arg11, %arg12] : memref<8x1xf32, strided<[1, 1], offset: ?>>
                    %18 = arith.mulf %14, %16 : f32
                    %19 = arith.mulf %15, %17 : f32
                    %20 = arith.subf %18, %19 : f32
                    %21 = arith.mulf %15, %16 : f32
                    %22 = arith.mulf %14, %17 : f32
                    %23 = arith.addf %21, %22 : f32
                    memref.store %20, %subview_105[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
                    memref.store %23, %subview_106[%arg9, %arg10, %arg11, %arg12] : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
                  }
                }
              }
            }
            memref.copy %subview_105, %subview_105 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
            memref.copy %subview_106, %subview_106 : memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>> to memref<1x?x8x1xf32, strided<[384, 32, 1, 1], offset: ?>>
          }
        }
        %alloc_72 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        %subview_73 = memref.subview %alloc_72[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        memref.copy %alloc_70, %subview_73 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %alloc_74 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        memref.copy %alloc_72, %alloc_74 : memref<1x12x32x2xf32> to memref<1x12x32x2xf32>
        %subview_75 = memref.subview %alloc_74[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        memref.copy %alloc_71, %subview_75 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape_76 = memref.collapse_shape %alloc_74 [[0], [1], [2, 3]] : memref<1x12x32x2xf32> into memref<1x12x64xf32>
        %reshape_77 = memref.reshape %collapse_shape_61(%2) : (memref<1x12x64xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %12 = arith.index_cast %arg0 : i64 to index
        %alloc_78 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
        memref.copy %arg5, %alloc_78 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
        %subview_79 = memref.subview %alloc_78[%arg3, %12, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape_77, %subview_79 : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        %reshape_80 = memref.reshape %collapse_shape_76(%2) : (memref<1x12x64xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %alloc_81 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
        memref.copy %arg6, %alloc_81 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
        %subview_82 = memref.subview %alloc_81[%arg3, %12, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape_80, %subview_82 : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_83 = memref.subview %alloc_78[%arg3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1024x768xf32, strided<[786432, 768, 1], offset: ?>>
        %alloc_84 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x768xf32>
        memref.copy %subview_83, %alloc_84 : memref<1x1024x768xf32, strided<[786432, 768, 1], offset: ?>> to memref<1x1024x768xf32>
        %reshape_85 = memref.reshape %alloc_84(%1) : (memref<1x1024x768xf32>, memref<2xi64>) -> memref<1024x768xf32>
        %subview_86 = memref.subview %alloc_81[%arg3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1024x768xf32, strided<[786432, 768, 1], offset: ?>>
        %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x768xf32>
        memref.copy %subview_86, %alloc_87 : memref<1x1024x768xf32, strided<[786432, 768, 1], offset: ?>> to memref<1x1024x768xf32>
        %reshape_88 = memref.reshape %alloc_87(%1) : (memref<1x1024x768xf32>, memref<2xi64>) -> memref<1024x768xf32>
        %alloc_89 = memref.alloc() {alignment = 64 : i64} : memref<1x12x64xf32>
        memref.copy %6, %alloc_89 : memref<1x12x64xf32> to memref<1x12x64xf32>
        scf.for %arg7 = %c0 to %c12 step %c1 {
          %13 = arith.index_cast %arg7 : index to i64
          %14 = arith.muli %13, %c64_i64 : i64
          %15 = arith.index_cast %14 : i64 to index
          %subview_101 = memref.subview %reshape_46[0, %15] [1, 64] [1, 1] : memref<1x768xf32> to memref<1x64xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %reshape_85[0, %15] [1024, 64] [1, 1] : memref<1024x768xf32> to memref<1024x64xf32, strided<[768, 1], offset: ?>>
          %alloc_103 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_103[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                memref.store %cst_0, %subview_113[%arg9, %arg10] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
              }
            }
            memref.copy %subview_113, %subview_113 : memref<1x8xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
          }
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            scf.for %arg9 = %c0 to %c64 step %c8 {
              %subview_113 = memref.subview %subview_101[0, %arg9] [1, 8] [1, 1] : memref<1x64xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
              %subview_114 = memref.subview %subview_102[%arg8, %arg9] [8, 8] [1, 1] : memref<1024x64xf32, strided<[768, 1], offset: ?>> to memref<8x8xf32, strided<[768, 1], offset: ?>>
              %subview_115 = memref.subview %alloc_103[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
              scf.for %arg10 = %c0 to %c1 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  scf.for %arg12 = %c0 to %c8 step %c1 {
                    %16 = memref.load %subview_113[%arg10, %arg12] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                    %17 = memref.load %subview_114[%arg11, %arg12] : memref<8x8xf32, strided<[768, 1], offset: ?>>
                    %18 = memref.load %subview_115[%arg10, %arg11] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
                    %19 = affine.apply #map(%arg11, %arg8)
                    %20 = arith.index_cast %19 : index to i64
                    %21 = arith.cmpi sle, %20, %arg0 : i64
                    %22 = arith.select %21, %cst, %cst_0 : f32
                    %23 = arith.mulf %16, %17 : f32
                    %24 = arith.addf %18, %23 : f32
                    %25 = arith.cmpf ugt, %22, %cst_7 : f32
                    %26 = arith.select %25, %24, %cst_6 : f32
                    memref.store %26, %subview_115[%arg10, %arg11] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
                  }
                }
              }
              memref.copy %subview_115, %subview_115 : memref<1x8xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            }
          }
          %alloc_104 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_103[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            %subview_114 = memref.subview %alloc_104[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                %16 = memref.load %subview_113[%arg9, %arg10] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
                %17 = arith.mulf %16, %cst_1 : f32
                memref.store %17, %subview_114[%arg9, %arg10] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
              }
            }
            memref.copy %subview_114, %subview_114 : memref<1x8xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
          }
          %alloc_105 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            memref.store %cst_16, %alloc_105[%arg8] : memref<1xf32>
          }
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_104[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                %16 = memref.load %subview_113[%arg9, %arg10] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
                %17 = memref.load %alloc_105[%arg9] : memref<1xf32>
                %18 = arith.maxnumf %16, %17 : f32
                memref.store %18, %alloc_105[%arg9] : memref<1xf32>
              }
            }
          }
          %alloc_106 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_104[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            %subview_114 = memref.subview %alloc_106[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                %16 = memref.load %subview_113[%arg9, %arg10] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
                %17 = memref.load %alloc_105[%arg9] : memref<1xf32>
                %18 = arith.subf %16, %17 : f32
                %19 = math.exp %18 : f32
                memref.store %19, %subview_114[%arg9, %arg10] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
              }
            }
            memref.copy %subview_114, %subview_114 : memref<1x8xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
          }
          %alloc_107 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            memref.store %cst_0, %alloc_107[%arg8] : memref<1xf32>
          }
          scf.for %arg8 = %c0 to %c1024 step %c8 {
            %subview_113 = memref.subview %alloc_106[0, %arg8] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                %16 = memref.load %subview_113[%arg9, %arg10] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
                %17 = memref.load %alloc_107[%arg9] : memref<1xf32>
                %18 = arith.addf %16, %17 : f32
                memref.store %18, %alloc_107[%arg9] : memref<1xf32>
              }
            }
          }
          %subview_108 = memref.subview %reshape_88[0, %15] [1024, 64] [1, 1] : memref<1024x768xf32> to memref<1024x64xf32, strided<[768, 1], offset: ?>>
          %alloc_109 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
          scf.for %arg8 = %c0 to %c64 step %c8 {
            %subview_113 = memref.subview %alloc_109[0, %arg8] [1, 8] [1, 1] : memref<1x64xf32> to memref<1x8xf32, strided<[64, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                memref.store %cst_0, %subview_113[%arg9, %arg10] : memref<1x8xf32, strided<[64, 1], offset: ?>>
              }
            }
            memref.copy %subview_113, %subview_113 : memref<1x8xf32, strided<[64, 1], offset: ?>> to memref<1x8xf32, strided<[64, 1], offset: ?>>
          }
          %alloc_110 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
          memref.copy %alloc_109, %alloc_110 : memref<1x64xf32> to memref<1x64xf32>
          scf.for %arg8 = %c0 to %c64 step %c8 {
            scf.for %arg9 = %c0 to %c1024 step %c8 {
              %subview_113 = memref.subview %alloc_106[0, %arg9] [1, 8] [1, 1] : memref<1x1024xf32> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
              %subview_114 = memref.subview %subview_108[%arg9, %arg8] [8, 8] [1, 1] : memref<1024x64xf32, strided<[768, 1], offset: ?>> to memref<8x8xf32, strided<[768, 1], offset: ?>>
              %subview_115 = memref.subview %alloc_110[0, %arg8] [1, 8] [1, 1] : memref<1x64xf32> to memref<1x8xf32, strided<[64, 1], offset: ?>>
              scf.for %arg10 = %c0 to %c1 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  scf.for %arg12 = %c0 to %c8 step %c1 {
                    %16 = memref.load %subview_113[%arg10, %arg12] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
                    %17 = memref.load %alloc_107[%arg10] : memref<1xf32>
                    %18 = memref.load %subview_114[%arg12, %arg11] : memref<8x8xf32, strided<[768, 1], offset: ?>>
                    %19 = memref.load %subview_115[%arg10, %arg11] : memref<1x8xf32, strided<[64, 1], offset: ?>>
                    %20 = arith.divf %16, %17 : f32
                    %21 = arith.mulf %20, %18 : f32
                    %22 = arith.addf %19, %21 : f32
                    memref.store %22, %subview_115[%arg10, %arg11] : memref<1x8xf32, strided<[64, 1], offset: ?>>
                  }
                }
              }
              memref.copy %subview_115, %subview_115 : memref<1x8xf32, strided<[64, 1], offset: ?>> to memref<1x8xf32, strided<[64, 1], offset: ?>>
            }
          }
          %reshape_111 = memref.reshape %alloc_110(%0) : (memref<1x64xf32>, memref<3xi64>) -> memref<1x1x64xf32>
          %subview_112 = memref.subview %alloc_89[0, %arg7, 0] [1, 1, 64] [1, 1, 1] : memref<1x12x64xf32> to memref<1x1x64xf32, strided<[768, 64, 1], offset: ?>>
          memref.copy %reshape_111, %subview_112 : memref<1x1x64xf32> to memref<1x1x64xf32, strided<[768, 64, 1], offset: ?>>
        }
        %reshape_90 = memref.reshape %alloc_89(%3) : (memref<1x12x64xf32>, memref<2xi64>) -> memref<1x768xf32>
        %alloc_91 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_91[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              memref.store %cst_0, %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %reshape_90[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_91[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  %13 = memref.load %subview_101[%arg9, %arg11] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %14 = memref.load %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %15 = arith.mulf %13, %cst_15 : f32
                  %16 = arith.addf %14, %15 : f32
                  memref.store %16, %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_92 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %arg4[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %alloc_91[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_103 = memref.subview %alloc_92[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              %13 = memref.load %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %14 = memref.load %subview_102[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %15 = arith.addf %13, %14 : f32
              memref.store %15, %subview_103[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_103, %subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_93 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        scf.for %arg7 = %c0 to %c1 step %c1 {
          memref.store %cst_0, %alloc_93[%arg7] : memref<1xf32>
        }
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_92[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              %13 = memref.load %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %14 = memref.load %alloc_93[%arg8] : memref<1xf32>
              %15 = arith.mulf %13, %13 : f32
              %16 = arith.addf %14, %15 : f32
              memref.store %16, %alloc_93[%arg8] : memref<1xf32>
            }
          }
        }
        %alloc_94 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_92[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %alloc_94[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              %13 = memref.load %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %14 = memref.load %alloc_93[%arg8] : memref<1xf32>
              %15 = arith.divf %14, %cst_8 : f32
              %16 = arith.addf %15, %cst_2 : f32
              %17 = math.rsqrt %16 : f32
              %18 = arith.mulf %13, %17 : f32
              %19 = arith.mulf %18, %cst_14 : f32
              memref.store %19, %subview_102[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<1x2048xf32>
        scf.for %arg7 = %c0 to %c2048 step %c8 {
          %subview_101 = memref.subview %alloc_95[0, %arg7] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              memref.store %cst_0, %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[2048, 1], offset: ?>>
            }
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[2048, 1], offset: ?>> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
        }
        scf.for %arg7 = %c0 to %c2048 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_94[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_95[0, %arg7] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  %13 = memref.load %subview_101[%arg9, %arg11] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %14 = memref.load %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[2048, 1], offset: ?>>
                  %15 = arith.mulf %13, %cst_13 : f32
                  %16 = arith.addf %14, %15 : f32
                  memref.store %16, %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[2048, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[2048, 1], offset: ?>> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
          }
        }
        %alloc_96 = memref.alloc() {alignment = 64 : i64} : memref<1x2048xf32>
        scf.for %arg7 = %c0 to %c2048 step %c8 {
          %subview_101 = memref.subview %alloc_96[0, %arg7] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              memref.store %cst_0, %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[2048, 1], offset: ?>>
            }
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[2048, 1], offset: ?>> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
        }
        scf.for %arg7 = %c0 to %c2048 step %c8 {
          scf.for %arg8 = %c0 to %c768 step %c8 {
            %subview_101 = memref.subview %alloc_94[0, %arg8] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_96[0, %arg7] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  %13 = memref.load %subview_101[%arg9, %arg11] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %14 = memref.load %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[2048, 1], offset: ?>>
                  %15 = arith.mulf %13, %cst_12 : f32
                  %16 = arith.addf %14, %15 : f32
                  memref.store %16, %subview_102[%arg9, %arg10] : memref<1x8xf32, strided<[2048, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_102, %subview_102 : memref<1x8xf32, strided<[2048, 1], offset: ?>> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
          }
        }
        %alloc_97 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_97[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              memref.store %cst_0, %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_101, %subview_101 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        scf.for %arg7 = %c0 to %c768 step %c8 {
          scf.for %arg8 = %c0 to %c2048 step %c8 {
            %subview_101 = memref.subview %alloc_95[0, %arg8] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
            %subview_102 = memref.subview %alloc_96[0, %arg8] [1, 8] [1, 1] : memref<1x2048xf32> to memref<1x8xf32, strided<[2048, 1], offset: ?>>
            %subview_103 = memref.subview %alloc_97[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            scf.for %arg9 = %c0 to %c1 step %c1 {
              scf.for %arg10 = %c0 to %c8 step %c1 {
                scf.for %arg11 = %c0 to %c8 step %c1 {
                  %13 = memref.load %subview_101[%arg9, %arg11] : memref<1x8xf32, strided<[2048, 1], offset: ?>>
                  %14 = memref.load %subview_102[%arg9, %arg11] : memref<1x8xf32, strided<[2048, 1], offset: ?>>
                  %15 = memref.load %subview_103[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %16 = arith.negf %13 : f32
                  %17 = math.exp %16 : f32
                  %18 = arith.addf %13, %17 : f32
                  %19 = arith.divf %13, %18 : f32
                  %20 = arith.mulf %19, %14 : f32
                  %21 = arith.mulf %20, %cst_11 : f32
                  %22 = arith.addf %15, %21 : f32
                  memref.store %22, %subview_103[%arg9, %arg10] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_103, %subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_98 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg7 = %c0 to %c768 step %c8 {
          %subview_101 = memref.subview %alloc_92[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_102 = memref.subview %alloc_97[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_103 = memref.subview %alloc_98[0, %arg7] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg8 = %c0 to %c1 step %c1 {
            scf.for %arg9 = %c0 to %c8 step %c1 {
              %13 = memref.load %subview_101[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %14 = memref.load %subview_102[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %15 = arith.addf %13, %14 : f32
              memref.store %15, %subview_103[%arg8, %arg9] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_103, %subview_103 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_99 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
        memref.copy %alloc_78, %alloc_99 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
        %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
        memref.copy %alloc_81, %alloc_100 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
        scf.yield %alloc_98, %alloc_99, %alloc_100 : memref<1x768xf32>, memref<12x1024x768xf32>, memref<12x1024x768xf32>
      }
      %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      scf.for %arg3 = %c0 to %c1 step %c1 {
        memref.store %cst_0, %alloc_23[%arg3] : memref<1xf32>
      }
      scf.for %arg3 = %c0 to %c768 step %c8 {
        %subview = memref.subview %9#0[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        scf.for %arg4 = %c0 to %c1 step %c1 {
          scf.for %arg5 = %c0 to %c8 step %c1 {
            %11 = memref.load %subview[%arg4, %arg5] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            %12 = memref.load %alloc_23[%arg4] : memref<1xf32>
            %13 = arith.mulf %11, %11 : f32
            %14 = arith.addf %12, %13 : f32
            memref.store %14, %alloc_23[%arg4] : memref<1xf32>
          }
        }
      }
      %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x32000xf32>
      scf.for %arg3 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_24[0, %arg3] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        scf.for %arg4 = %c0 to %c1 step %c1 {
          scf.for %arg5 = %c0 to %c8 step %c1 {
            memref.store %cst_0, %subview[%arg4, %arg5] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
          }
        }
        memref.copy %subview, %subview : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
      }
      scf.for %arg3 = %c0 to %c32000 step %c8 {
        scf.for %arg4 = %c0 to %c768 step %c8 {
          %subview = memref.subview %9#0[0, %arg4] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_27 = memref.subview %alloc_24[0, %arg3] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c1 step %c1 {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %11 = memref.load %subview[%arg5, %arg7] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                %12 = memref.load %alloc_23[%arg5] : memref<1xf32>
                %13 = memref.load %subview_27[%arg5, %arg6] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
                %14 = arith.divf %12, %cst_8 : f32
                %15 = arith.addf %14, %cst_2 : f32
                %16 = math.rsqrt %15 : f32
                %17 = arith.mulf %11, %16 : f32
                %18 = arith.mulf %17, %cst_10 : f32
                %19 = arith.mulf %18, %cst_9 : f32
                %20 = arith.addf %13, %19 : f32
                memref.store %20, %subview_27[%arg5, %arg6] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
              }
            }
          }
          memref.copy %subview_27, %subview_27 : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        }
      }
      %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      scf.for %arg3 = %c0 to %c1 step %c1 {
        memref.store %cst_16, %alloc_25[%arg3] : memref<1xf32>
      }
      %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
      scf.for %arg3 = %c0 to %c1 step %c1 {
        memref.store %c0_i64, %alloc_26[%arg3] : memref<1xi64>
      }
      scf.for %arg3 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_24[0, %arg3] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        scf.for %arg4 = %c0 to %c1 step %c1 {
          scf.for %arg5 = %c0 to %c8 step %c1 {
            %11 = memref.load %subview[%arg4, %arg5] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
            %12 = memref.load %alloc_25[%arg4] : memref<1xf32>
            %13 = memref.load %alloc_26[%arg4] : memref<1xi64>
            %14 = affine.apply #map(%arg5, %arg3)
            %15 = arith.index_cast %14 : index to i64
            %16 = arith.cmpf ogt, %11, %12 : f32
            %17 = arith.select %16, %11, %12 : f32
            %18 = arith.select %16, %15, %13 : i64
            memref.store %17, %alloc_25[%arg4] : memref<1xf32>
            memref.store %18, %alloc_26[%arg4] : memref<1xi64>
          }
        }
      }
      %cast = memref.cast %alloc_24 : memref<1x32000xf32> to memref<*xf32>
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
      %10 = arith.addi %arg0, %c1_i64 : i64
      scf.yield %10, %9#1, %9#2 : i64, memref<12x1024x768xf32>, memref<12x1024x768xf32>
    }
    return
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_12x1024x768xf32(dense<0.000000e+00> : tensor<12x1024x768xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<12 x array<1024 x array<768 x f32>>>
  llvm.mlir.global private constant @__constant_1x12x64xf32(dense<0.000000e+00> : tensor<1x12x64xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<12 x array<64 x f32>>>
  llvm.mlir.global private constant @__constant_1x768xf32(dense<2.000000e+00> : tensor<1x768xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<768 x f32>>
  llvm.mlir.global private constant @__constant_3xi64_1(dense<[1, 12, 64]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x i64>
  llvm.mlir.global private constant @__constant_2xi64_0(dense<[1, 768]> : tensor<2xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x i64>
  llvm.mlir.global private constant @__constant_3xi64_0(dense<[1, 1, 768]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x i64>
  llvm.mlir.global private constant @__constant_2xi64(dense<[1024, 768]> : tensor<2xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x i64>
  llvm.mlir.global private constant @__constant_3xi64(dense<[1, 1, 64]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x i64>
  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @host() {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.mlir.constant(-1 : index) : i64
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %4 = llvm.mlir.constant(384 : index) : i64
    %5 = llvm.mlir.addressof @__constant_12x1024x768xf32 : !llvm.ptr
    %6 = llvm.mlir.constant(786432 : index) : i64
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %8 = llvm.mlir.addressof @__constant_1x12x64xf32 : !llvm.ptr
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.mlir.addressof @__constant_1x768xf32 : !llvm.ptr
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(10 : i64) : i64
    %13 = llvm.mlir.constant(0 : i64) : i64
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.mlir.constant(12 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %19 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %20 = llvm.mlir.constant(64 : i64) : i64
    %21 = llvm.mlir.constant(1.250000e-01 : f32) : f32
    %22 = llvm.mlir.constant(9.99999974E-6 : f32) : f32
    %23 = llvm.mlir.constant(1.000000e+04 : f32) : f32
    %24 = llvm.mlir.constant(6.400000e+01 : f32) : f32
    %25 = llvm.mlir.constant(-2.000000e+00 : f32) : f32
    %26 = llvm.mlir.constant(-1.000000e+09 : f32) : f32
    %27 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %28 = llvm.mlir.constant(7.680000e+02 : f32) : f32
    %29 = llvm.mlir.constant(32000 : index) : i64
    %30 = llvm.mlir.constant(2048 : index) : i64
    %31 = llvm.mlir.constant(64 : index) : i64
    %32 = llvm.mlir.constant(1024 : index) : i64
    %33 = llvm.mlir.constant(32 : index) : i64
    %34 = llvm.mlir.constant(768 : index) : i64
    %35 = llvm.mlir.constant(8 : index) : i64
    %36 = llvm.mlir.constant(1.300000e+01 : f32) : f32
    %37 = llvm.mlir.constant(1.200000e+01 : f32) : f32
    %38 = llvm.mlir.constant(1.000000e+01 : f32) : f32
    %39 = llvm.mlir.constant(1.100000e+01 : f32) : f32
    %40 = llvm.mlir.constant(9.000000e+00 : f32) : f32
    %41 = llvm.mlir.constant(8.000000e+00 : f32) : f32
    %42 = llvm.mlir.constant(7.000000e+00 : f32) : f32
    %43 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %44 = llvm.mlir.constant(6.000000e+00 : f32) : f32
    %45 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %46 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %47 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %48 = llvm.mlir.zero : !llvm.ptr
    %49 = llvm.getelementptr %10[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x f32>>
    %50 = llvm.getelementptr %8[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<12 x array<64 x f32>>>
    %51 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12 x array<1024 x array<768 x f32>>>
    %52 = llvm.getelementptr %48[9437184] : (!llvm.ptr) -> !llvm.ptr, f32
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.add %53, %31 : i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.sub %31, %16 : i64
    %58 = llvm.add %56, %57 : i64
    %59 = llvm.urem %58, %31  : i64
    %60 = llvm.sub %58, %59 : i64
    %61 = llvm.inttoptr %60 : i64 to !llvm.ptr
    %62 = llvm.insertvalue %55, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %63 = llvm.insertvalue %61, %62[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %64 = llvm.insertvalue %17, %63[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %65 = llvm.insertvalue %15, %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.insertvalue %32, %65[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.insertvalue %34, %66[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = llvm.insertvalue %6, %67[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.insertvalue %34, %68[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.insertvalue %16, %69[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.mul %15, %16 : i64
    %72 = llvm.mul %71, %32 : i64
    %73 = llvm.mul %72, %34 : i64
    %74 = llvm.getelementptr %48[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %75 = llvm.ptrtoint %74 : !llvm.ptr to i64
    %76 = llvm.mul %73, %75 : i64
    "llvm.intr.memcpy"(%61, %51, %76) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %77 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %78 = llvm.ptrtoint %77 : !llvm.ptr to i64
    %79 = llvm.add %78, %57 : i64
    %80 = llvm.urem %79, %31  : i64
    %81 = llvm.sub %79, %80 : i64
    %82 = llvm.inttoptr %81 : i64 to !llvm.ptr
    %83 = llvm.insertvalue %77, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %84 = llvm.insertvalue %82, %83[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %85 = llvm.insertvalue %17, %84[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %86 = llvm.insertvalue %15, %85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %87 = llvm.insertvalue %32, %86[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %88 = llvm.insertvalue %34, %87[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %89 = llvm.insertvalue %6, %88[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %90 = llvm.insertvalue %34, %89[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %91 = llvm.insertvalue %16, %90[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    "llvm.intr.memcpy"(%82, %51, %76) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%13, %70, %91 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb1(%92: i64, %93: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, %94: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // 2 preds: ^bb0, ^bb449
    %95 = llvm.icmp "slt" %92, %12 : i64
    %96 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    %98 = llvm.add %97, %57 : i64
    %99 = llvm.urem %98, %31  : i64
    %100 = llvm.sub %98, %99 : i64
    %101 = llvm.inttoptr %100 : i64 to !llvm.ptr
    %102 = llvm.insertvalue %96, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %103 = llvm.insertvalue %101, %102[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %104 = llvm.insertvalue %17, %103[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %105 = llvm.insertvalue %15, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %106 = llvm.insertvalue %32, %105[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %107 = llvm.insertvalue %34, %106[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %108 = llvm.insertvalue %6, %107[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %109 = llvm.insertvalue %34, %108[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %110 = llvm.insertvalue %16, %109[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %111 = llvm.extractvalue %93[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %112 = llvm.mul %111, %16 : i64
    %113 = llvm.extractvalue %93[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %114 = llvm.mul %112, %113 : i64
    %115 = llvm.extractvalue %93[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %116 = llvm.mul %114, %115 : i64
    %117 = llvm.mul %116, %75 : i64
    %118 = llvm.extractvalue %93[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %119 = llvm.extractvalue %93[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %120 = llvm.getelementptr %118[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%101, %120, %117) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %121 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %122 = llvm.ptrtoint %121 : !llvm.ptr to i64
    %123 = llvm.add %122, %57 : i64
    %124 = llvm.urem %123, %31  : i64
    %125 = llvm.sub %123, %124 : i64
    %126 = llvm.inttoptr %125 : i64 to !llvm.ptr
    %127 = llvm.insertvalue %121, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %128 = llvm.insertvalue %126, %127[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %129 = llvm.insertvalue %17, %128[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %130 = llvm.insertvalue %15, %129[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %131 = llvm.insertvalue %32, %130[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %132 = llvm.insertvalue %34, %131[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %133 = llvm.insertvalue %6, %132[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %134 = llvm.insertvalue %34, %133[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %135 = llvm.insertvalue %16, %134[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %136 = llvm.extractvalue %94[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %137 = llvm.mul %136, %16 : i64
    %138 = llvm.extractvalue %94[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %139 = llvm.mul %137, %138 : i64
    %140 = llvm.extractvalue %94[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %141 = llvm.mul %139, %140 : i64
    %142 = llvm.mul %141, %75 : i64
    %143 = llvm.extractvalue %94[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %144 = llvm.extractvalue %94[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %145 = llvm.getelementptr %143[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%126, %145, %142) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.cond_br %95, ^bb2(%92, %110, %135 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>), ^bb450
  ^bb2(%146: i64, %147: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, %148: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // pred: ^bb1
    %149 = llvm.getelementptr %48[768] : (!llvm.ptr) -> !llvm.ptr, f32
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    %151 = llvm.add %150, %31 : i64
    %152 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %153 = llvm.ptrtoint %152 : !llvm.ptr to i64
    %154 = llvm.add %153, %57 : i64
    %155 = llvm.urem %154, %31  : i64
    %156 = llvm.sub %154, %155 : i64
    %157 = llvm.inttoptr %156 : i64 to !llvm.ptr
    %158 = llvm.insertvalue %152, %9[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %157, %158[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.insertvalue %17, %159[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %16, %160[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %34, %161[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %34, %162[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %16, %163[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.mul %16, %16 : i64
    %166 = llvm.mul %165, %34 : i64
    %167 = llvm.mul %166, %75 : i64
    "llvm.intr.memcpy"(%157, %49, %167) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%17, %164, %147, %148 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb3(%168: i64, %169: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %170: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, %171: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // 2 preds: ^bb2, ^bb397
    %172 = llvm.icmp "slt" %168, %15 : i64
    llvm.cond_br %172, ^bb4, ^bb398
  ^bb4:  // pred: ^bb3
    %173 = llvm.add %75, %31 : i64
    %174 = llvm.call @malloc(%173) : (i64) -> !llvm.ptr
    %175 = llvm.ptrtoint %174 : !llvm.ptr to i64
    %176 = llvm.add %175, %57 : i64
    %177 = llvm.urem %176, %31  : i64
    %178 = llvm.sub %176, %177 : i64
    %179 = llvm.inttoptr %178 : i64 to !llvm.ptr
    llvm.br ^bb5(%17 : i64)
  ^bb5(%180: i64):  // 2 preds: ^bb4, ^bb6
    %181 = llvm.icmp "slt" %180, %16 : i64
    llvm.cond_br %181, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %182 = llvm.getelementptr %179[%180] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %182 : f32, !llvm.ptr
    %183 = llvm.add %180, %16 : i64
    llvm.br ^bb5(%183 : i64)
  ^bb7:  // pred: ^bb5
    llvm.br ^bb8(%17 : i64)
  ^bb8(%184: i64):  // 2 preds: ^bb7, ^bb15
    %185 = llvm.icmp "slt" %184, %34 : i64
    llvm.cond_br %185, ^bb9, ^bb16
  ^bb9:  // pred: ^bb8
    %186 = llvm.extractvalue %169[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb10(%17 : i64)
  ^bb10(%187: i64):  // 2 preds: ^bb9, ^bb14
    %188 = llvm.icmp "slt" %187, %16 : i64
    llvm.cond_br %188, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12(%17 : i64)
  ^bb12(%189: i64):  // 2 preds: ^bb11, ^bb13
    %190 = llvm.icmp "slt" %189, %35 : i64
    llvm.cond_br %190, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %191 = llvm.getelementptr %186[%184] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %192 = llvm.mul %187, %34 : i64
    %193 = llvm.add %192, %189 : i64
    %194 = llvm.getelementptr %191[%193] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %195 = llvm.load %194 : !llvm.ptr -> f32
    %196 = llvm.getelementptr %179[%187] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %197 = llvm.load %196 : !llvm.ptr -> f32
    %198 = llvm.fmul %195, %195  : f32
    %199 = llvm.fadd %197, %198  : f32
    llvm.store %199, %196 : f32, !llvm.ptr
    %200 = llvm.add %189, %16 : i64
    llvm.br ^bb12(%200 : i64)
  ^bb14:  // pred: ^bb12
    %201 = llvm.add %187, %16 : i64
    llvm.br ^bb10(%201 : i64)
  ^bb15:  // pred: ^bb10
    %202 = llvm.add %184, %35 : i64
    llvm.br ^bb8(%202 : i64)
  ^bb16:  // pred: ^bb8
    %203 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %204 = llvm.ptrtoint %203 : !llvm.ptr to i64
    %205 = llvm.add %204, %57 : i64
    %206 = llvm.urem %205, %31  : i64
    %207 = llvm.sub %205, %206 : i64
    %208 = llvm.inttoptr %207 : i64 to !llvm.ptr
    llvm.br ^bb17(%17 : i64)
  ^bb17(%209: i64):  // 2 preds: ^bb16, ^bb24
    %210 = llvm.icmp "slt" %209, %34 : i64
    llvm.cond_br %210, ^bb18, ^bb25
  ^bb18:  // pred: ^bb17
    %211 = llvm.extractvalue %169[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb19(%17 : i64)
  ^bb19(%212: i64):  // 2 preds: ^bb18, ^bb23
    %213 = llvm.icmp "slt" %212, %16 : i64
    llvm.cond_br %213, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%17 : i64)
  ^bb21(%214: i64):  // 2 preds: ^bb20, ^bb22
    %215 = llvm.icmp "slt" %214, %35 : i64
    llvm.cond_br %215, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %216 = llvm.getelementptr %211[%209] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %217 = llvm.mul %212, %34 : i64
    %218 = llvm.add %217, %214 : i64
    %219 = llvm.getelementptr %216[%218] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %220 = llvm.load %219 : !llvm.ptr -> f32
    %221 = llvm.getelementptr %179[%212] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %222 = llvm.load %221 : !llvm.ptr -> f32
    %223 = llvm.fdiv %222, %28  : f32
    %224 = llvm.fadd %223, %22  : f32
    %225 = llvm.intr.sqrt(%224)  : (f32) -> f32
    %226 = llvm.fdiv %18, %225  : f32
    %227 = llvm.fmul %220, %226  : f32
    %228 = llvm.fmul %227, %47  : f32
    %229 = llvm.getelementptr %208[%209] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %230 = llvm.getelementptr %229[%218] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %228, %230 : f32, !llvm.ptr
    %231 = llvm.add %214, %16 : i64
    llvm.br ^bb21(%231 : i64)
  ^bb23:  // pred: ^bb21
    %232 = llvm.add %212, %16 : i64
    llvm.br ^bb19(%232 : i64)
  ^bb24:  // pred: ^bb19
    %233 = llvm.mul %165, %35 : i64
    %234 = llvm.mul %233, %75 : i64
    %235 = llvm.getelementptr %208[%209] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%235, %235, %234) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %236 = llvm.add %209, %35 : i64
    llvm.br ^bb17(%236 : i64)
  ^bb25:  // pred: ^bb17
    %237 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %238 = llvm.ptrtoint %237 : !llvm.ptr to i64
    %239 = llvm.add %238, %57 : i64
    %240 = llvm.urem %239, %31  : i64
    %241 = llvm.sub %239, %240 : i64
    %242 = llvm.inttoptr %241 : i64 to !llvm.ptr
    llvm.br ^bb26(%17 : i64)
  ^bb26(%243: i64):  // 2 preds: ^bb25, ^bb33
    %244 = llvm.icmp "slt" %243, %34 : i64
    llvm.cond_br %244, ^bb27, ^bb34
  ^bb27:  // pred: ^bb26
    llvm.br ^bb28(%17 : i64)
  ^bb28(%245: i64):  // 2 preds: ^bb27, ^bb32
    %246 = llvm.icmp "slt" %245, %16 : i64
    llvm.cond_br %246, ^bb29, ^bb33
  ^bb29:  // pred: ^bb28
    llvm.br ^bb30(%17 : i64)
  ^bb30(%247: i64):  // 2 preds: ^bb29, ^bb31
    %248 = llvm.icmp "slt" %247, %35 : i64
    llvm.cond_br %248, ^bb31, ^bb32
  ^bb31:  // pred: ^bb30
    %249 = llvm.getelementptr %242[%243] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %250 = llvm.mul %245, %34 : i64
    %251 = llvm.add %250, %247 : i64
    %252 = llvm.getelementptr %249[%251] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %252 : f32, !llvm.ptr
    %253 = llvm.add %247, %16 : i64
    llvm.br ^bb30(%253 : i64)
  ^bb32:  // pred: ^bb30
    %254 = llvm.add %245, %16 : i64
    llvm.br ^bb28(%254 : i64)
  ^bb33:  // pred: ^bb28
    %255 = llvm.mul %165, %35 : i64
    %256 = llvm.mul %255, %75 : i64
    %257 = llvm.getelementptr %242[%243] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%257, %257, %256) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %258 = llvm.add %243, %35 : i64
    llvm.br ^bb26(%258 : i64)
  ^bb34:  // pred: ^bb26
    %259 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %260 = llvm.ptrtoint %259 : !llvm.ptr to i64
    %261 = llvm.add %260, %57 : i64
    %262 = llvm.urem %261, %31  : i64
    %263 = llvm.sub %261, %262 : i64
    %264 = llvm.inttoptr %263 : i64 to !llvm.ptr
    "llvm.intr.memcpy"(%264, %242, %167) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb35(%17 : i64)
  ^bb35(%265: i64):  // 2 preds: ^bb34, ^bb48
    %266 = llvm.icmp "slt" %265, %34 : i64
    llvm.cond_br %266, ^bb36, ^bb49
  ^bb36:  // pred: ^bb35
    llvm.br ^bb37(%17 : i64)
  ^bb37(%267: i64):  // 2 preds: ^bb36, ^bb47
    %268 = llvm.icmp "slt" %267, %34 : i64
    llvm.cond_br %268, ^bb38, ^bb48
  ^bb38:  // pred: ^bb37
    llvm.br ^bb39(%17 : i64)
  ^bb39(%269: i64):  // 2 preds: ^bb38, ^bb46
    %270 = llvm.icmp "slt" %269, %16 : i64
    llvm.cond_br %270, ^bb40, ^bb47
  ^bb40:  // pred: ^bb39
    llvm.br ^bb41(%17 : i64)
  ^bb41(%271: i64):  // 2 preds: ^bb40, ^bb45
    %272 = llvm.icmp "slt" %271, %35 : i64
    llvm.cond_br %272, ^bb42, ^bb46
  ^bb42:  // pred: ^bb41
    llvm.br ^bb43(%17 : i64)
  ^bb43(%273: i64):  // 2 preds: ^bb42, ^bb44
    %274 = llvm.icmp "slt" %273, %35 : i64
    llvm.cond_br %274, ^bb44, ^bb45
  ^bb44:  // pred: ^bb43
    %275 = llvm.getelementptr %208[%267] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %276 = llvm.mul %269, %34 : i64
    %277 = llvm.add %276, %273 : i64
    %278 = llvm.getelementptr %275[%277] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %279 = llvm.load %278 : !llvm.ptr -> f32
    %280 = llvm.getelementptr %264[%265] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %281 = llvm.add %276, %271 : i64
    %282 = llvm.getelementptr %280[%281] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %283 = llvm.load %282 : !llvm.ptr -> f32
    %284 = llvm.fmul %279, %46  : f32
    %285 = llvm.fadd %283, %284  : f32
    llvm.store %285, %282 : f32, !llvm.ptr
    %286 = llvm.add %273, %16 : i64
    llvm.br ^bb43(%286 : i64)
  ^bb45:  // pred: ^bb43
    %287 = llvm.add %271, %16 : i64
    llvm.br ^bb41(%287 : i64)
  ^bb46:  // pred: ^bb41
    %288 = llvm.add %269, %16 : i64
    llvm.br ^bb39(%288 : i64)
  ^bb47:  // pred: ^bb39
    %289 = llvm.mul %165, %35 : i64
    %290 = llvm.mul %289, %75 : i64
    %291 = llvm.getelementptr %264[%265] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%291, %291, %290) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %292 = llvm.add %267, %35 : i64
    llvm.br ^bb37(%292 : i64)
  ^bb48:  // pred: ^bb37
    %293 = llvm.add %265, %35 : i64
    llvm.br ^bb35(%293 : i64)
  ^bb49:  // pred: ^bb35
    %294 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %295 = llvm.ptrtoint %294 : !llvm.ptr to i64
    %296 = llvm.add %295, %57 : i64
    %297 = llvm.urem %296, %31  : i64
    %298 = llvm.sub %296, %297 : i64
    %299 = llvm.inttoptr %298 : i64 to !llvm.ptr
    llvm.br ^bb50(%17 : i64)
  ^bb50(%300: i64):  // 2 preds: ^bb49, ^bb57
    %301 = llvm.icmp "slt" %300, %34 : i64
    llvm.cond_br %301, ^bb51, ^bb58
  ^bb51:  // pred: ^bb50
    llvm.br ^bb52(%17 : i64)
  ^bb52(%302: i64):  // 2 preds: ^bb51, ^bb56
    %303 = llvm.icmp "slt" %302, %16 : i64
    llvm.cond_br %303, ^bb53, ^bb57
  ^bb53:  // pred: ^bb52
    llvm.br ^bb54(%17 : i64)
  ^bb54(%304: i64):  // 2 preds: ^bb53, ^bb55
    %305 = llvm.icmp "slt" %304, %35 : i64
    llvm.cond_br %305, ^bb55, ^bb56
  ^bb55:  // pred: ^bb54
    %306 = llvm.getelementptr %299[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %307 = llvm.mul %302, %34 : i64
    %308 = llvm.add %307, %304 : i64
    %309 = llvm.getelementptr %306[%308] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %309 : f32, !llvm.ptr
    %310 = llvm.add %304, %16 : i64
    llvm.br ^bb54(%310 : i64)
  ^bb56:  // pred: ^bb54
    %311 = llvm.add %302, %16 : i64
    llvm.br ^bb52(%311 : i64)
  ^bb57:  // pred: ^bb52
    %312 = llvm.mul %165, %35 : i64
    %313 = llvm.mul %312, %75 : i64
    %314 = llvm.getelementptr %299[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%314, %314, %313) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %315 = llvm.add %300, %35 : i64
    llvm.br ^bb50(%315 : i64)
  ^bb58:  // pred: ^bb50
    %316 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %317 = llvm.ptrtoint %316 : !llvm.ptr to i64
    %318 = llvm.add %317, %57 : i64
    %319 = llvm.urem %318, %31  : i64
    %320 = llvm.sub %318, %319 : i64
    %321 = llvm.inttoptr %320 : i64 to !llvm.ptr
    "llvm.intr.memcpy"(%321, %299, %167) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb59(%17 : i64)
  ^bb59(%322: i64):  // 2 preds: ^bb58, ^bb72
    %323 = llvm.icmp "slt" %322, %34 : i64
    llvm.cond_br %323, ^bb60, ^bb73
  ^bb60:  // pred: ^bb59
    llvm.br ^bb61(%17 : i64)
  ^bb61(%324: i64):  // 2 preds: ^bb60, ^bb71
    %325 = llvm.icmp "slt" %324, %34 : i64
    llvm.cond_br %325, ^bb62, ^bb72
  ^bb62:  // pred: ^bb61
    llvm.br ^bb63(%17 : i64)
  ^bb63(%326: i64):  // 2 preds: ^bb62, ^bb70
    %327 = llvm.icmp "slt" %326, %16 : i64
    llvm.cond_br %327, ^bb64, ^bb71
  ^bb64:  // pred: ^bb63
    llvm.br ^bb65(%17 : i64)
  ^bb65(%328: i64):  // 2 preds: ^bb64, ^bb69
    %329 = llvm.icmp "slt" %328, %35 : i64
    llvm.cond_br %329, ^bb66, ^bb70
  ^bb66:  // pred: ^bb65
    llvm.br ^bb67(%17 : i64)
  ^bb67(%330: i64):  // 2 preds: ^bb66, ^bb68
    %331 = llvm.icmp "slt" %330, %35 : i64
    llvm.cond_br %331, ^bb68, ^bb69
  ^bb68:  // pred: ^bb67
    %332 = llvm.getelementptr %208[%324] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %333 = llvm.mul %326, %34 : i64
    %334 = llvm.add %333, %330 : i64
    %335 = llvm.getelementptr %332[%334] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %336 = llvm.load %335 : !llvm.ptr -> f32
    %337 = llvm.getelementptr %321[%322] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %338 = llvm.add %333, %328 : i64
    %339 = llvm.getelementptr %337[%338] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %340 = llvm.load %339 : !llvm.ptr -> f32
    %341 = llvm.fmul %336, %45  : f32
    %342 = llvm.fadd %340, %341  : f32
    llvm.store %342, %339 : f32, !llvm.ptr
    %343 = llvm.add %330, %16 : i64
    llvm.br ^bb67(%343 : i64)
  ^bb69:  // pred: ^bb67
    %344 = llvm.add %328, %16 : i64
    llvm.br ^bb65(%344 : i64)
  ^bb70:  // pred: ^bb65
    %345 = llvm.add %326, %16 : i64
    llvm.br ^bb63(%345 : i64)
  ^bb71:  // pred: ^bb63
    %346 = llvm.mul %165, %35 : i64
    %347 = llvm.mul %346, %75 : i64
    %348 = llvm.getelementptr %321[%322] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%348, %348, %347) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %349 = llvm.add %324, %35 : i64
    llvm.br ^bb61(%349 : i64)
  ^bb72:  // pred: ^bb61
    %350 = llvm.add %322, %35 : i64
    llvm.br ^bb59(%350 : i64)
  ^bb73:  // pred: ^bb59
    %351 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %352 = llvm.ptrtoint %351 : !llvm.ptr to i64
    %353 = llvm.add %352, %57 : i64
    %354 = llvm.urem %353, %31  : i64
    %355 = llvm.sub %353, %354 : i64
    %356 = llvm.inttoptr %355 : i64 to !llvm.ptr
    llvm.br ^bb74(%17 : i64)
  ^bb74(%357: i64):  // 2 preds: ^bb73, ^bb81
    %358 = llvm.icmp "slt" %357, %34 : i64
    llvm.cond_br %358, ^bb75, ^bb82
  ^bb75:  // pred: ^bb74
    llvm.br ^bb76(%17 : i64)
  ^bb76(%359: i64):  // 2 preds: ^bb75, ^bb80
    %360 = llvm.icmp "slt" %359, %16 : i64
    llvm.cond_br %360, ^bb77, ^bb81
  ^bb77:  // pred: ^bb76
    llvm.br ^bb78(%17 : i64)
  ^bb78(%361: i64):  // 2 preds: ^bb77, ^bb79
    %362 = llvm.icmp "slt" %361, %35 : i64
    llvm.cond_br %362, ^bb79, ^bb80
  ^bb79:  // pred: ^bb78
    %363 = llvm.getelementptr %356[%357] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %364 = llvm.mul %359, %34 : i64
    %365 = llvm.add %364, %361 : i64
    %366 = llvm.getelementptr %363[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %366 : f32, !llvm.ptr
    %367 = llvm.add %361, %16 : i64
    llvm.br ^bb78(%367 : i64)
  ^bb80:  // pred: ^bb78
    %368 = llvm.add %359, %16 : i64
    llvm.br ^bb76(%368 : i64)
  ^bb81:  // pred: ^bb76
    %369 = llvm.mul %165, %35 : i64
    %370 = llvm.mul %369, %75 : i64
    %371 = llvm.getelementptr %356[%357] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%371, %371, %370) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %372 = llvm.add %357, %35 : i64
    llvm.br ^bb74(%372 : i64)
  ^bb82:  // pred: ^bb74
    %373 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %374 = llvm.ptrtoint %373 : !llvm.ptr to i64
    %375 = llvm.add %374, %57 : i64
    %376 = llvm.urem %375, %31  : i64
    %377 = llvm.sub %375, %376 : i64
    %378 = llvm.inttoptr %377 : i64 to !llvm.ptr
    "llvm.intr.memcpy"(%378, %356, %167) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb83(%17 : i64)
  ^bb83(%379: i64):  // 2 preds: ^bb82, ^bb96
    %380 = llvm.icmp "slt" %379, %34 : i64
    llvm.cond_br %380, ^bb84, ^bb97
  ^bb84:  // pred: ^bb83
    llvm.br ^bb85(%17 : i64)
  ^bb85(%381: i64):  // 2 preds: ^bb84, ^bb95
    %382 = llvm.icmp "slt" %381, %34 : i64
    llvm.cond_br %382, ^bb86, ^bb96
  ^bb86:  // pred: ^bb85
    llvm.br ^bb87(%17 : i64)
  ^bb87(%383: i64):  // 2 preds: ^bb86, ^bb94
    %384 = llvm.icmp "slt" %383, %16 : i64
    llvm.cond_br %384, ^bb88, ^bb95
  ^bb88:  // pred: ^bb87
    llvm.br ^bb89(%17 : i64)
  ^bb89(%385: i64):  // 2 preds: ^bb88, ^bb93
    %386 = llvm.icmp "slt" %385, %35 : i64
    llvm.cond_br %386, ^bb90, ^bb94
  ^bb90:  // pred: ^bb89
    llvm.br ^bb91(%17 : i64)
  ^bb91(%387: i64):  // 2 preds: ^bb90, ^bb92
    %388 = llvm.icmp "slt" %387, %35 : i64
    llvm.cond_br %388, ^bb92, ^bb93
  ^bb92:  // pred: ^bb91
    %389 = llvm.getelementptr %208[%381] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %390 = llvm.mul %383, %34 : i64
    %391 = llvm.add %390, %387 : i64
    %392 = llvm.getelementptr %389[%391] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %393 = llvm.load %392 : !llvm.ptr -> f32
    %394 = llvm.getelementptr %378[%379] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %395 = llvm.add %390, %385 : i64
    %396 = llvm.getelementptr %394[%395] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %397 = llvm.load %396 : !llvm.ptr -> f32
    %398 = llvm.fmul %393, %44  : f32
    %399 = llvm.fadd %397, %398  : f32
    llvm.store %399, %396 : f32, !llvm.ptr
    %400 = llvm.add %387, %16 : i64
    llvm.br ^bb91(%400 : i64)
  ^bb93:  // pred: ^bb91
    %401 = llvm.add %385, %16 : i64
    llvm.br ^bb89(%401 : i64)
  ^bb94:  // pred: ^bb89
    %402 = llvm.add %383, %16 : i64
    llvm.br ^bb87(%402 : i64)
  ^bb95:  // pred: ^bb87
    %403 = llvm.mul %165, %35 : i64
    %404 = llvm.mul %403, %75 : i64
    %405 = llvm.getelementptr %378[%379] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%405, %405, %404) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %406 = llvm.add %381, %35 : i64
    llvm.br ^bb85(%406 : i64)
  ^bb96:  // pred: ^bb85
    %407 = llvm.add %379, %35 : i64
    llvm.br ^bb83(%407 : i64)
  ^bb97:  // pred: ^bb83
    %408 = llvm.getelementptr %48[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %409 = llvm.ptrtoint %408 : !llvm.ptr to i64
    %410 = llvm.add %409, %31 : i64
    %411 = llvm.call @malloc(%410) : (i64) -> !llvm.ptr
    %412 = llvm.ptrtoint %411 : !llvm.ptr to i64
    %413 = llvm.add %412, %57 : i64
    %414 = llvm.urem %413, %31  : i64
    %415 = llvm.sub %413, %414 : i64
    %416 = llvm.inttoptr %415 : i64 to !llvm.ptr
    %417 = llvm.call @malloc(%410) : (i64) -> !llvm.ptr
    %418 = llvm.ptrtoint %417 : !llvm.ptr to i64
    %419 = llvm.add %418, %57 : i64
    %420 = llvm.urem %419, %31  : i64
    %421 = llvm.sub %419, %420 : i64
    %422 = llvm.inttoptr %421 : i64 to !llvm.ptr
    %423 = llvm.uitofp %146 : i64 to f32
    llvm.br ^bb98(%17 : i64)
  ^bb98(%424: i64):  // 2 preds: ^bb97, ^bb102
    %425 = llvm.icmp "slt" %424, %33 : i64
    llvm.cond_br %425, ^bb99, ^bb103
  ^bb99:  // pred: ^bb98
    llvm.br ^bb100(%17 : i64)
  ^bb100(%426: i64):  // 2 preds: ^bb99, ^bb101
    %427 = llvm.icmp "slt" %426, %35 : i64
    llvm.cond_br %427, ^bb101, ^bb102
  ^bb101:  // pred: ^bb100
    %428 = llvm.add %426, %424 : i64
    %429 = llvm.uitofp %428 : i64 to f32
    %430 = llvm.fmul %429, %25  : f32
    %431 = llvm.fdiv %430, %24  : f32
    %432 = llvm.intr.pow(%23, %431)  : (f32, f32) -> f32
    %433 = llvm.fmul %423, %432  : f32
    %434 = llvm.intr.cos(%433)  : (f32) -> f32
    %435 = llvm.intr.sin(%433)  : (f32) -> f32
    %436 = llvm.getelementptr %416[%424] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %437 = llvm.getelementptr %436[%426] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %434, %437 : f32, !llvm.ptr
    %438 = llvm.getelementptr %422[%424] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %439 = llvm.getelementptr %438[%426] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %435, %439 : f32, !llvm.ptr
    %440 = llvm.add %426, %16 : i64
    llvm.br ^bb100(%440 : i64)
  ^bb102:  // pred: ^bb100
    %441 = llvm.mul %35, %16 : i64
    %442 = llvm.mul %441, %75 : i64
    %443 = llvm.getelementptr %416[%424] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%443, %443, %442) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %444 = llvm.getelementptr %422[%424] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%444, %444, %442) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %445 = llvm.add %424, %35 : i64
    llvm.br ^bb98(%445 : i64)
  ^bb103:  // pred: ^bb98
    %446 = llvm.getelementptr %48[384] : (!llvm.ptr) -> !llvm.ptr, f32
    %447 = llvm.ptrtoint %446 : !llvm.ptr to i64
    %448 = llvm.add %447, %31 : i64
    %449 = llvm.call @malloc(%448) : (i64) -> !llvm.ptr
    %450 = llvm.ptrtoint %449 : !llvm.ptr to i64
    %451 = llvm.add %450, %57 : i64
    %452 = llvm.urem %451, %31  : i64
    %453 = llvm.sub %451, %452 : i64
    %454 = llvm.inttoptr %453 : i64 to !llvm.ptr
    %455 = llvm.insertvalue %449, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %456 = llvm.insertvalue %454, %455[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %457 = llvm.insertvalue %17, %456[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %458 = llvm.insertvalue %16, %457[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %459 = llvm.insertvalue %15, %458[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %460 = llvm.insertvalue %33, %459[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %461 = llvm.insertvalue %16, %460[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %462 = llvm.insertvalue %4, %461[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %463 = llvm.insertvalue %33, %462[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %464 = llvm.insertvalue %16, %463[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %465 = llvm.insertvalue %16, %464[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %466 = llvm.call @malloc(%448) : (i64) -> !llvm.ptr
    %467 = llvm.ptrtoint %466 : !llvm.ptr to i64
    %468 = llvm.add %467, %57 : i64
    %469 = llvm.urem %468, %31  : i64
    %470 = llvm.sub %468, %469 : i64
    %471 = llvm.inttoptr %470 : i64 to !llvm.ptr
    %472 = llvm.insertvalue %466, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %473 = llvm.insertvalue %471, %472[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %474 = llvm.insertvalue %17, %473[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %475 = llvm.insertvalue %16, %474[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %476 = llvm.insertvalue %15, %475[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %477 = llvm.insertvalue %33, %476[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %478 = llvm.insertvalue %16, %477[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %479 = llvm.insertvalue %4, %478[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %480 = llvm.insertvalue %33, %479[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %481 = llvm.insertvalue %16, %480[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %482 = llvm.insertvalue %16, %481[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb104(%17 : i64)
  ^bb104(%483: i64):  // 2 preds: ^bb103, ^bb120
    %484 = llvm.icmp "slt" %483, %15 : i64
    llvm.cond_br %484, ^bb105, ^bb121
  ^bb105:  // pred: ^bb104
    llvm.br ^bb106(%17 : i64)
  ^bb106(%485: i64):  // 2 preds: ^bb105, ^bb119
    %486 = llvm.icmp "slt" %485, %33 : i64
    llvm.cond_br %486, ^bb107, ^bb120
  ^bb107:  // pred: ^bb106
    %487 = llvm.mul %483, %2 : i64
    %488 = llvm.add %487, %15 : i64
    %489 = llvm.intr.smin(%488, %35)  : (i64, i64) -> i64
    %490 = llvm.mul %483, %31 : i64
    %491 = llvm.mul %485, %11 : i64
    %492 = llvm.add %490, %491 : i64
    %493 = llvm.add %492, %16 : i64
    %494 = llvm.mul %483, %33 : i64
    %495 = llvm.add %494, %485 : i64
    %496 = llvm.insertvalue %495, %456[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %497 = llvm.insertvalue %16, %496[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %498 = llvm.insertvalue %4, %497[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %499 = llvm.insertvalue %489, %498[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %500 = llvm.insertvalue %33, %499[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %501 = llvm.insertvalue %35, %500[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %502 = llvm.insertvalue %16, %501[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %503 = llvm.insertvalue %16, %502[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %504 = llvm.insertvalue %16, %503[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %505 = llvm.insertvalue %495, %473[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %506 = llvm.insertvalue %16, %505[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %507 = llvm.insertvalue %4, %506[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %508 = llvm.insertvalue %489, %507[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %509 = llvm.insertvalue %33, %508[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %510 = llvm.insertvalue %35, %509[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %511 = llvm.insertvalue %16, %510[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %512 = llvm.insertvalue %16, %511[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %513 = llvm.insertvalue %16, %512[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb108(%17 : i64)
  ^bb108(%514: i64):  // 2 preds: ^bb107, ^bb118
    %515 = llvm.icmp "slt" %514, %16 : i64
    llvm.cond_br %515, ^bb109, ^bb119
  ^bb109:  // pred: ^bb108
    llvm.br ^bb110(%17 : i64)
  ^bb110(%516: i64):  // 2 preds: ^bb109, ^bb117
    %517 = llvm.icmp "slt" %516, %489 : i64
    llvm.cond_br %517, ^bb111, ^bb118
  ^bb111:  // pred: ^bb110
    llvm.br ^bb112(%17 : i64)
  ^bb112(%518: i64):  // 2 preds: ^bb111, ^bb116
    %519 = llvm.icmp "slt" %518, %35 : i64
    llvm.cond_br %519, ^bb113, ^bb117
  ^bb113:  // pred: ^bb112
    llvm.br ^bb114(%17 : i64)
  ^bb114(%520: i64):  // 2 preds: ^bb113, ^bb115
    %521 = llvm.icmp "slt" %520, %16 : i64
    llvm.cond_br %521, ^bb115, ^bb116
  ^bb115:  // pred: ^bb114
    %522 = llvm.getelementptr %264[%492] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %523 = llvm.mul %514, %34 : i64
    %524 = llvm.mul %516, %31 : i64
    %525 = llvm.add %523, %524 : i64
    %526 = llvm.mul %518, %11 : i64
    %527 = llvm.add %525, %526 : i64
    %528 = llvm.add %527, %520 : i64
    %529 = llvm.getelementptr %522[%528] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %530 = llvm.load %529 : !llvm.ptr -> f32
    %531 = llvm.getelementptr %264[%493] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %532 = llvm.getelementptr %531[%528] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %533 = llvm.load %532 : !llvm.ptr -> f32
    %534 = llvm.getelementptr %416[%485] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %535 = llvm.add %518, %520 : i64
    %536 = llvm.getelementptr %534[%535] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %537 = llvm.load %536 : !llvm.ptr -> f32
    %538 = llvm.getelementptr %422[%485] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %539 = llvm.getelementptr %538[%535] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %540 = llvm.load %539 : !llvm.ptr -> f32
    %541 = llvm.fmul %530, %537  : f32
    %542 = llvm.fmul %533, %540  : f32
    %543 = llvm.fsub %541, %542  : f32
    %544 = llvm.fmul %533, %537  : f32
    %545 = llvm.fmul %530, %540  : f32
    %546 = llvm.fadd %544, %545  : f32
    %547 = llvm.getelementptr %454[%495] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %548 = llvm.mul %514, %4 : i64
    %549 = llvm.mul %516, %33 : i64
    %550 = llvm.add %548, %549 : i64
    %551 = llvm.add %550, %518 : i64
    %552 = llvm.add %551, %520 : i64
    %553 = llvm.getelementptr %547[%552] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %543, %553 : f32, !llvm.ptr
    %554 = llvm.getelementptr %471[%495] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %555 = llvm.getelementptr %554[%552] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %546, %555 : f32, !llvm.ptr
    %556 = llvm.add %520, %16 : i64
    llvm.br ^bb114(%556 : i64)
  ^bb116:  // pred: ^bb114
    %557 = llvm.add %518, %16 : i64
    llvm.br ^bb112(%557 : i64)
  ^bb117:  // pred: ^bb112
    %558 = llvm.add %516, %16 : i64
    llvm.br ^bb110(%558 : i64)
  ^bb118:  // pred: ^bb110
    %559 = llvm.add %514, %16 : i64
    llvm.br ^bb108(%559 : i64)
  ^bb119:  // pred: ^bb108
    %560 = llvm.intr.stacksave : !llvm.ptr
    %561 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %504, %561 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %562 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i64, ptr)> 
    %563 = llvm.insertvalue %561, %562[1] : !llvm.struct<(i64, ptr)> 
    %564 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %504, %564 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %565 = llvm.insertvalue %564, %562[1] : !llvm.struct<(i64, ptr)> 
    %566 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %563, %566 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %567 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %565, %567 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %566, %567) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %560 : !llvm.ptr
    %568 = llvm.intr.stacksave : !llvm.ptr
    %569 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %513, %569 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %570 = llvm.insertvalue %569, %562[1] : !llvm.struct<(i64, ptr)> 
    %571 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %513, %571 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %572 = llvm.insertvalue %571, %562[1] : !llvm.struct<(i64, ptr)> 
    %573 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %570, %573 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %574 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %572, %574 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %573, %574) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %568 : !llvm.ptr
    %575 = llvm.add %485, %35 : i64
    llvm.br ^bb106(%575 : i64)
  ^bb120:  // pred: ^bb106
    %576 = llvm.add %483, %35 : i64
    llvm.br ^bb104(%576 : i64)
  ^bb121:  // pred: ^bb104
    %577 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %578 = llvm.ptrtoint %577 : !llvm.ptr to i64
    %579 = llvm.add %578, %57 : i64
    %580 = llvm.urem %579, %31  : i64
    %581 = llvm.sub %579, %580 : i64
    %582 = llvm.inttoptr %581 : i64 to !llvm.ptr
    %583 = llvm.insertvalue %577, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %584 = llvm.insertvalue %582, %583[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %585 = llvm.insertvalue %17, %584[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %586 = llvm.insertvalue %16, %585[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %587 = llvm.insertvalue %34, %586[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %588 = llvm.insertvalue %15, %587[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %589 = llvm.insertvalue %31, %588[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %590 = llvm.insertvalue %33, %589[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %591 = llvm.insertvalue %11, %590[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %592 = llvm.insertvalue %16, %591[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %593 = llvm.insertvalue %16, %592[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %594 = llvm.intr.stacksave : !llvm.ptr
    %595 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %465, %595 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %596 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i64, ptr)> 
    %597 = llvm.insertvalue %595, %596[1] : !llvm.struct<(i64, ptr)> 
    %598 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %593, %598 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %599 = llvm.insertvalue %598, %596[1] : !llvm.struct<(i64, ptr)> 
    %600 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %597, %600 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %601 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %599, %601 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %600, %601) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %594 : !llvm.ptr
    %602 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %603 = llvm.ptrtoint %602 : !llvm.ptr to i64
    %604 = llvm.add %603, %57 : i64
    %605 = llvm.urem %604, %31  : i64
    %606 = llvm.sub %604, %605 : i64
    %607 = llvm.inttoptr %606 : i64 to !llvm.ptr
    %608 = llvm.mul %165, %15 : i64
    %609 = llvm.mul %608, %33 : i64
    %610 = llvm.mul %609, %11 : i64
    %611 = llvm.mul %610, %75 : i64
    "llvm.intr.memcpy"(%607, %582, %611) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %612 = llvm.insertvalue %602, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %613 = llvm.insertvalue %607, %612[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %614 = llvm.insertvalue %16, %613[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %615 = llvm.insertvalue %16, %614[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %616 = llvm.insertvalue %34, %615[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %617 = llvm.insertvalue %15, %616[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %618 = llvm.insertvalue %31, %617[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %619 = llvm.insertvalue %33, %618[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %620 = llvm.insertvalue %11, %619[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %621 = llvm.insertvalue %16, %620[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %622 = llvm.insertvalue %16, %621[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %623 = llvm.intr.stacksave : !llvm.ptr
    %624 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %482, %624 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %625 = llvm.insertvalue %624, %596[1] : !llvm.struct<(i64, ptr)> 
    %626 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %622, %626 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %627 = llvm.insertvalue %626, %596[1] : !llvm.struct<(i64, ptr)> 
    %628 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %625, %628 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %629 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %627, %629 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %628, %629) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %623 : !llvm.ptr
    %630 = llvm.call @malloc(%410) : (i64) -> !llvm.ptr
    %631 = llvm.ptrtoint %630 : !llvm.ptr to i64
    %632 = llvm.add %631, %57 : i64
    %633 = llvm.urem %632, %31  : i64
    %634 = llvm.sub %632, %633 : i64
    %635 = llvm.inttoptr %634 : i64 to !llvm.ptr
    %636 = llvm.call @malloc(%410) : (i64) -> !llvm.ptr
    %637 = llvm.ptrtoint %636 : !llvm.ptr to i64
    %638 = llvm.add %637, %57 : i64
    %639 = llvm.urem %638, %31  : i64
    %640 = llvm.sub %638, %639 : i64
    %641 = llvm.inttoptr %640 : i64 to !llvm.ptr
    llvm.br ^bb122(%17 : i64)
  ^bb122(%642: i64):  // 2 preds: ^bb121, ^bb126
    %643 = llvm.icmp "slt" %642, %33 : i64
    llvm.cond_br %643, ^bb123, ^bb127
  ^bb123:  // pred: ^bb122
    llvm.br ^bb124(%17 : i64)
  ^bb124(%644: i64):  // 2 preds: ^bb123, ^bb125
    %645 = llvm.icmp "slt" %644, %35 : i64
    llvm.cond_br %645, ^bb125, ^bb126
  ^bb125:  // pred: ^bb124
    %646 = llvm.add %644, %642 : i64
    %647 = llvm.uitofp %646 : i64 to f32
    %648 = llvm.fmul %647, %25  : f32
    %649 = llvm.fdiv %648, %24  : f32
    %650 = llvm.intr.pow(%23, %649)  : (f32, f32) -> f32
    %651 = llvm.fmul %423, %650  : f32
    %652 = llvm.intr.cos(%651)  : (f32) -> f32
    %653 = llvm.intr.sin(%651)  : (f32) -> f32
    %654 = llvm.getelementptr %635[%642] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %655 = llvm.getelementptr %654[%644] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %652, %655 : f32, !llvm.ptr
    %656 = llvm.getelementptr %641[%642] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %657 = llvm.getelementptr %656[%644] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %653, %657 : f32, !llvm.ptr
    %658 = llvm.add %644, %16 : i64
    llvm.br ^bb124(%658 : i64)
  ^bb126:  // pred: ^bb124
    %659 = llvm.mul %35, %16 : i64
    %660 = llvm.mul %659, %75 : i64
    %661 = llvm.getelementptr %635[%642] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%661, %661, %660) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %662 = llvm.getelementptr %641[%642] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%662, %662, %660) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %663 = llvm.add %642, %35 : i64
    llvm.br ^bb122(%663 : i64)
  ^bb127:  // pred: ^bb122
    %664 = llvm.call @malloc(%448) : (i64) -> !llvm.ptr
    %665 = llvm.ptrtoint %664 : !llvm.ptr to i64
    %666 = llvm.add %665, %57 : i64
    %667 = llvm.urem %666, %31  : i64
    %668 = llvm.sub %666, %667 : i64
    %669 = llvm.inttoptr %668 : i64 to !llvm.ptr
    %670 = llvm.insertvalue %664, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %671 = llvm.insertvalue %669, %670[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %672 = llvm.insertvalue %17, %671[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %673 = llvm.insertvalue %16, %672[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %674 = llvm.insertvalue %15, %673[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %675 = llvm.insertvalue %33, %674[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %676 = llvm.insertvalue %16, %675[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %677 = llvm.insertvalue %4, %676[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %678 = llvm.insertvalue %33, %677[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %679 = llvm.insertvalue %16, %678[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %680 = llvm.insertvalue %16, %679[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %681 = llvm.call @malloc(%448) : (i64) -> !llvm.ptr
    %682 = llvm.ptrtoint %681 : !llvm.ptr to i64
    %683 = llvm.add %682, %57 : i64
    %684 = llvm.urem %683, %31  : i64
    %685 = llvm.sub %683, %684 : i64
    %686 = llvm.inttoptr %685 : i64 to !llvm.ptr
    %687 = llvm.insertvalue %681, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %688 = llvm.insertvalue %686, %687[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %689 = llvm.insertvalue %17, %688[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %690 = llvm.insertvalue %16, %689[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %691 = llvm.insertvalue %15, %690[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %692 = llvm.insertvalue %33, %691[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %693 = llvm.insertvalue %16, %692[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %694 = llvm.insertvalue %4, %693[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %695 = llvm.insertvalue %33, %694[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %696 = llvm.insertvalue %16, %695[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %697 = llvm.insertvalue %16, %696[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb128(%17 : i64)
  ^bb128(%698: i64):  // 2 preds: ^bb127, ^bb144
    %699 = llvm.icmp "slt" %698, %15 : i64
    llvm.cond_br %699, ^bb129, ^bb145
  ^bb129:  // pred: ^bb128
    llvm.br ^bb130(%17 : i64)
  ^bb130(%700: i64):  // 2 preds: ^bb129, ^bb143
    %701 = llvm.icmp "slt" %700, %33 : i64
    llvm.cond_br %701, ^bb131, ^bb144
  ^bb131:  // pred: ^bb130
    %702 = llvm.mul %698, %2 : i64
    %703 = llvm.add %702, %15 : i64
    %704 = llvm.intr.smin(%703, %35)  : (i64, i64) -> i64
    %705 = llvm.mul %698, %31 : i64
    %706 = llvm.mul %700, %11 : i64
    %707 = llvm.add %705, %706 : i64
    %708 = llvm.add %707, %16 : i64
    %709 = llvm.mul %698, %33 : i64
    %710 = llvm.add %709, %700 : i64
    %711 = llvm.insertvalue %710, %671[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %712 = llvm.insertvalue %16, %711[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %713 = llvm.insertvalue %4, %712[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %714 = llvm.insertvalue %704, %713[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %715 = llvm.insertvalue %33, %714[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %716 = llvm.insertvalue %35, %715[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %717 = llvm.insertvalue %16, %716[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %718 = llvm.insertvalue %16, %717[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %719 = llvm.insertvalue %16, %718[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %720 = llvm.insertvalue %710, %688[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %721 = llvm.insertvalue %16, %720[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %722 = llvm.insertvalue %4, %721[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %723 = llvm.insertvalue %704, %722[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %724 = llvm.insertvalue %33, %723[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %725 = llvm.insertvalue %35, %724[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %726 = llvm.insertvalue %16, %725[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %727 = llvm.insertvalue %16, %726[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %728 = llvm.insertvalue %16, %727[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb132(%17 : i64)
  ^bb132(%729: i64):  // 2 preds: ^bb131, ^bb142
    %730 = llvm.icmp "slt" %729, %16 : i64
    llvm.cond_br %730, ^bb133, ^bb143
  ^bb133:  // pred: ^bb132
    llvm.br ^bb134(%17 : i64)
  ^bb134(%731: i64):  // 2 preds: ^bb133, ^bb141
    %732 = llvm.icmp "slt" %731, %704 : i64
    llvm.cond_br %732, ^bb135, ^bb142
  ^bb135:  // pred: ^bb134
    llvm.br ^bb136(%17 : i64)
  ^bb136(%733: i64):  // 2 preds: ^bb135, ^bb140
    %734 = llvm.icmp "slt" %733, %35 : i64
    llvm.cond_br %734, ^bb137, ^bb141
  ^bb137:  // pred: ^bb136
    llvm.br ^bb138(%17 : i64)
  ^bb138(%735: i64):  // 2 preds: ^bb137, ^bb139
    %736 = llvm.icmp "slt" %735, %16 : i64
    llvm.cond_br %736, ^bb139, ^bb140
  ^bb139:  // pred: ^bb138
    %737 = llvm.getelementptr %321[%707] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %738 = llvm.mul %729, %34 : i64
    %739 = llvm.mul %731, %31 : i64
    %740 = llvm.add %738, %739 : i64
    %741 = llvm.mul %733, %11 : i64
    %742 = llvm.add %740, %741 : i64
    %743 = llvm.add %742, %735 : i64
    %744 = llvm.getelementptr %737[%743] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %745 = llvm.load %744 : !llvm.ptr -> f32
    %746 = llvm.getelementptr %321[%708] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %747 = llvm.getelementptr %746[%743] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %748 = llvm.load %747 : !llvm.ptr -> f32
    %749 = llvm.getelementptr %635[%700] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %750 = llvm.add %733, %735 : i64
    %751 = llvm.getelementptr %749[%750] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %752 = llvm.load %751 : !llvm.ptr -> f32
    %753 = llvm.getelementptr %641[%700] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %754 = llvm.getelementptr %753[%750] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %755 = llvm.load %754 : !llvm.ptr -> f32
    %756 = llvm.fmul %745, %752  : f32
    %757 = llvm.fmul %748, %755  : f32
    %758 = llvm.fsub %756, %757  : f32
    %759 = llvm.fmul %748, %752  : f32
    %760 = llvm.fmul %745, %755  : f32
    %761 = llvm.fadd %759, %760  : f32
    %762 = llvm.getelementptr %669[%710] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %763 = llvm.mul %729, %4 : i64
    %764 = llvm.mul %731, %33 : i64
    %765 = llvm.add %763, %764 : i64
    %766 = llvm.add %765, %733 : i64
    %767 = llvm.add %766, %735 : i64
    %768 = llvm.getelementptr %762[%767] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %758, %768 : f32, !llvm.ptr
    %769 = llvm.getelementptr %686[%710] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %770 = llvm.getelementptr %769[%767] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %761, %770 : f32, !llvm.ptr
    %771 = llvm.add %735, %16 : i64
    llvm.br ^bb138(%771 : i64)
  ^bb140:  // pred: ^bb138
    %772 = llvm.add %733, %16 : i64
    llvm.br ^bb136(%772 : i64)
  ^bb141:  // pred: ^bb136
    %773 = llvm.add %731, %16 : i64
    llvm.br ^bb134(%773 : i64)
  ^bb142:  // pred: ^bb134
    %774 = llvm.add %729, %16 : i64
    llvm.br ^bb132(%774 : i64)
  ^bb143:  // pred: ^bb132
    %775 = llvm.intr.stacksave : !llvm.ptr
    %776 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %719, %776 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %777 = llvm.insertvalue %776, %596[1] : !llvm.struct<(i64, ptr)> 
    %778 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %719, %778 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %779 = llvm.insertvalue %778, %596[1] : !llvm.struct<(i64, ptr)> 
    %780 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %777, %780 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %781 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %779, %781 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %780, %781) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %775 : !llvm.ptr
    %782 = llvm.intr.stacksave : !llvm.ptr
    %783 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %728, %783 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %784 = llvm.insertvalue %783, %596[1] : !llvm.struct<(i64, ptr)> 
    %785 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %728, %785 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %786 = llvm.insertvalue %785, %596[1] : !llvm.struct<(i64, ptr)> 
    %787 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %784, %787 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %788 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %786, %788 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %787, %788) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %782 : !llvm.ptr
    %789 = llvm.add %700, %35 : i64
    llvm.br ^bb130(%789 : i64)
  ^bb144:  // pred: ^bb130
    %790 = llvm.add %698, %35 : i64
    llvm.br ^bb128(%790 : i64)
  ^bb145:  // pred: ^bb128
    %791 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %792 = llvm.ptrtoint %791 : !llvm.ptr to i64
    %793 = llvm.add %792, %57 : i64
    %794 = llvm.urem %793, %31  : i64
    %795 = llvm.sub %793, %794 : i64
    %796 = llvm.inttoptr %795 : i64 to !llvm.ptr
    %797 = llvm.insertvalue %791, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %798 = llvm.insertvalue %796, %797[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %799 = llvm.insertvalue %17, %798[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %800 = llvm.insertvalue %16, %799[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %801 = llvm.insertvalue %34, %800[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %802 = llvm.insertvalue %15, %801[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %803 = llvm.insertvalue %31, %802[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %804 = llvm.insertvalue %33, %803[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %805 = llvm.insertvalue %11, %804[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %806 = llvm.insertvalue %16, %805[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %807 = llvm.insertvalue %16, %806[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %808 = llvm.intr.stacksave : !llvm.ptr
    %809 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %680, %809 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %810 = llvm.insertvalue %809, %596[1] : !llvm.struct<(i64, ptr)> 
    %811 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %807, %811 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %812 = llvm.insertvalue %811, %596[1] : !llvm.struct<(i64, ptr)> 
    %813 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %810, %813 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %814 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %812, %814 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %813, %814) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %808 : !llvm.ptr
    %815 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %816 = llvm.ptrtoint %815 : !llvm.ptr to i64
    %817 = llvm.add %816, %57 : i64
    %818 = llvm.urem %817, %31  : i64
    %819 = llvm.sub %817, %818 : i64
    %820 = llvm.inttoptr %819 : i64 to !llvm.ptr
    "llvm.intr.memcpy"(%820, %796, %611) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %821 = llvm.insertvalue %815, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %822 = llvm.insertvalue %820, %821[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %823 = llvm.insertvalue %16, %822[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %824 = llvm.insertvalue %16, %823[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %825 = llvm.insertvalue %34, %824[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %826 = llvm.insertvalue %15, %825[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %827 = llvm.insertvalue %31, %826[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %828 = llvm.insertvalue %33, %827[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %829 = llvm.insertvalue %11, %828[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %830 = llvm.insertvalue %16, %829[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %831 = llvm.insertvalue %16, %830[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %832 = llvm.intr.stacksave : !llvm.ptr
    %833 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %697, %833 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %834 = llvm.insertvalue %833, %596[1] : !llvm.struct<(i64, ptr)> 
    %835 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %831, %835 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %836 = llvm.insertvalue %835, %596[1] : !llvm.struct<(i64, ptr)> 
    %837 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %834, %837 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %838 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %836, %838 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %837, %838) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %832 : !llvm.ptr
    %839 = llvm.call @malloc(%410) : (i64) -> !llvm.ptr
    %840 = llvm.ptrtoint %839 : !llvm.ptr to i64
    %841 = llvm.add %840, %57 : i64
    %842 = llvm.urem %841, %31  : i64
    %843 = llvm.sub %841, %842 : i64
    %844 = llvm.inttoptr %843 : i64 to !llvm.ptr
    %845 = llvm.call @malloc(%410) : (i64) -> !llvm.ptr
    %846 = llvm.ptrtoint %845 : !llvm.ptr to i64
    %847 = llvm.add %846, %57 : i64
    %848 = llvm.urem %847, %31  : i64
    %849 = llvm.sub %847, %848 : i64
    %850 = llvm.inttoptr %849 : i64 to !llvm.ptr
    llvm.br ^bb146(%17 : i64)
  ^bb146(%851: i64):  // 2 preds: ^bb145, ^bb150
    %852 = llvm.icmp "slt" %851, %33 : i64
    llvm.cond_br %852, ^bb147, ^bb151
  ^bb147:  // pred: ^bb146
    llvm.br ^bb148(%17 : i64)
  ^bb148(%853: i64):  // 2 preds: ^bb147, ^bb149
    %854 = llvm.icmp "slt" %853, %35 : i64
    llvm.cond_br %854, ^bb149, ^bb150
  ^bb149:  // pred: ^bb148
    %855 = llvm.add %853, %851 : i64
    %856 = llvm.uitofp %855 : i64 to f32
    %857 = llvm.fmul %856, %25  : f32
    %858 = llvm.fdiv %857, %24  : f32
    %859 = llvm.intr.pow(%23, %858)  : (f32, f32) -> f32
    %860 = llvm.fmul %423, %859  : f32
    %861 = llvm.intr.cos(%860)  : (f32) -> f32
    %862 = llvm.intr.sin(%860)  : (f32) -> f32
    %863 = llvm.getelementptr %844[%851] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %864 = llvm.getelementptr %863[%853] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %861, %864 : f32, !llvm.ptr
    %865 = llvm.getelementptr %850[%851] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %866 = llvm.getelementptr %865[%853] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %862, %866 : f32, !llvm.ptr
    %867 = llvm.add %853, %16 : i64
    llvm.br ^bb148(%867 : i64)
  ^bb150:  // pred: ^bb148
    %868 = llvm.mul %35, %16 : i64
    %869 = llvm.mul %868, %75 : i64
    %870 = llvm.getelementptr %844[%851] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%870, %870, %869) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %871 = llvm.getelementptr %850[%851] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%871, %871, %869) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %872 = llvm.add %851, %35 : i64
    llvm.br ^bb146(%872 : i64)
  ^bb151:  // pred: ^bb146
    %873 = llvm.call @malloc(%448) : (i64) -> !llvm.ptr
    %874 = llvm.ptrtoint %873 : !llvm.ptr to i64
    %875 = llvm.add %874, %57 : i64
    %876 = llvm.urem %875, %31  : i64
    %877 = llvm.sub %875, %876 : i64
    %878 = llvm.inttoptr %877 : i64 to !llvm.ptr
    %879 = llvm.insertvalue %873, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %880 = llvm.insertvalue %878, %879[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %881 = llvm.insertvalue %17, %880[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %882 = llvm.insertvalue %16, %881[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %883 = llvm.insertvalue %15, %882[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %884 = llvm.insertvalue %33, %883[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %885 = llvm.insertvalue %16, %884[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %886 = llvm.insertvalue %4, %885[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %887 = llvm.insertvalue %33, %886[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %888 = llvm.insertvalue %16, %887[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %889 = llvm.insertvalue %16, %888[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %890 = llvm.call @malloc(%448) : (i64) -> !llvm.ptr
    %891 = llvm.ptrtoint %890 : !llvm.ptr to i64
    %892 = llvm.add %891, %57 : i64
    %893 = llvm.urem %892, %31  : i64
    %894 = llvm.sub %892, %893 : i64
    %895 = llvm.inttoptr %894 : i64 to !llvm.ptr
    %896 = llvm.insertvalue %890, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %897 = llvm.insertvalue %895, %896[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %898 = llvm.insertvalue %17, %897[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %899 = llvm.insertvalue %16, %898[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %900 = llvm.insertvalue %15, %899[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %901 = llvm.insertvalue %33, %900[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %902 = llvm.insertvalue %16, %901[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %903 = llvm.insertvalue %4, %902[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %904 = llvm.insertvalue %33, %903[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %905 = llvm.insertvalue %16, %904[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %906 = llvm.insertvalue %16, %905[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb152(%17 : i64)
  ^bb152(%907: i64):  // 2 preds: ^bb151, ^bb168
    %908 = llvm.icmp "slt" %907, %15 : i64
    llvm.cond_br %908, ^bb153, ^bb169
  ^bb153:  // pred: ^bb152
    llvm.br ^bb154(%17 : i64)
  ^bb154(%909: i64):  // 2 preds: ^bb153, ^bb167
    %910 = llvm.icmp "slt" %909, %33 : i64
    llvm.cond_br %910, ^bb155, ^bb168
  ^bb155:  // pred: ^bb154
    %911 = llvm.mul %907, %2 : i64
    %912 = llvm.add %911, %15 : i64
    %913 = llvm.intr.smin(%912, %35)  : (i64, i64) -> i64
    %914 = llvm.mul %907, %31 : i64
    %915 = llvm.mul %909, %11 : i64
    %916 = llvm.add %914, %915 : i64
    %917 = llvm.add %916, %16 : i64
    %918 = llvm.mul %907, %33 : i64
    %919 = llvm.add %918, %909 : i64
    %920 = llvm.insertvalue %919, %880[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %921 = llvm.insertvalue %16, %920[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %922 = llvm.insertvalue %4, %921[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %923 = llvm.insertvalue %913, %922[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %924 = llvm.insertvalue %33, %923[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %925 = llvm.insertvalue %35, %924[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %926 = llvm.insertvalue %16, %925[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %927 = llvm.insertvalue %16, %926[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %928 = llvm.insertvalue %16, %927[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %929 = llvm.insertvalue %919, %897[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %930 = llvm.insertvalue %16, %929[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %931 = llvm.insertvalue %4, %930[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %932 = llvm.insertvalue %913, %931[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %933 = llvm.insertvalue %33, %932[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %934 = llvm.insertvalue %35, %933[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %935 = llvm.insertvalue %16, %934[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %936 = llvm.insertvalue %16, %935[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %937 = llvm.insertvalue %16, %936[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb156(%17 : i64)
  ^bb156(%938: i64):  // 2 preds: ^bb155, ^bb166
    %939 = llvm.icmp "slt" %938, %16 : i64
    llvm.cond_br %939, ^bb157, ^bb167
  ^bb157:  // pred: ^bb156
    llvm.br ^bb158(%17 : i64)
  ^bb158(%940: i64):  // 2 preds: ^bb157, ^bb165
    %941 = llvm.icmp "slt" %940, %913 : i64
    llvm.cond_br %941, ^bb159, ^bb166
  ^bb159:  // pred: ^bb158
    llvm.br ^bb160(%17 : i64)
  ^bb160(%942: i64):  // 2 preds: ^bb159, ^bb164
    %943 = llvm.icmp "slt" %942, %35 : i64
    llvm.cond_br %943, ^bb161, ^bb165
  ^bb161:  // pred: ^bb160
    llvm.br ^bb162(%17 : i64)
  ^bb162(%944: i64):  // 2 preds: ^bb161, ^bb163
    %945 = llvm.icmp "slt" %944, %16 : i64
    llvm.cond_br %945, ^bb163, ^bb164
  ^bb163:  // pred: ^bb162
    %946 = llvm.getelementptr %378[%916] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %947 = llvm.mul %938, %34 : i64
    %948 = llvm.mul %940, %31 : i64
    %949 = llvm.add %947, %948 : i64
    %950 = llvm.mul %942, %11 : i64
    %951 = llvm.add %949, %950 : i64
    %952 = llvm.add %951, %944 : i64
    %953 = llvm.getelementptr %946[%952] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %954 = llvm.load %953 : !llvm.ptr -> f32
    %955 = llvm.getelementptr %378[%917] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %956 = llvm.getelementptr %955[%952] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %957 = llvm.load %956 : !llvm.ptr -> f32
    %958 = llvm.getelementptr %844[%909] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %959 = llvm.add %942, %944 : i64
    %960 = llvm.getelementptr %958[%959] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %961 = llvm.load %960 : !llvm.ptr -> f32
    %962 = llvm.getelementptr %850[%909] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %963 = llvm.getelementptr %962[%959] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %964 = llvm.load %963 : !llvm.ptr -> f32
    %965 = llvm.fmul %954, %961  : f32
    %966 = llvm.fmul %957, %964  : f32
    %967 = llvm.fsub %965, %966  : f32
    %968 = llvm.fmul %957, %961  : f32
    %969 = llvm.fmul %954, %964  : f32
    %970 = llvm.fadd %968, %969  : f32
    %971 = llvm.getelementptr %878[%919] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %972 = llvm.mul %938, %4 : i64
    %973 = llvm.mul %940, %33 : i64
    %974 = llvm.add %972, %973 : i64
    %975 = llvm.add %974, %942 : i64
    %976 = llvm.add %975, %944 : i64
    %977 = llvm.getelementptr %971[%976] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %967, %977 : f32, !llvm.ptr
    %978 = llvm.getelementptr %895[%919] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %979 = llvm.getelementptr %978[%976] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %970, %979 : f32, !llvm.ptr
    %980 = llvm.add %944, %16 : i64
    llvm.br ^bb162(%980 : i64)
  ^bb164:  // pred: ^bb162
    %981 = llvm.add %942, %16 : i64
    llvm.br ^bb160(%981 : i64)
  ^bb165:  // pred: ^bb160
    %982 = llvm.add %940, %16 : i64
    llvm.br ^bb158(%982 : i64)
  ^bb166:  // pred: ^bb158
    %983 = llvm.add %938, %16 : i64
    llvm.br ^bb156(%983 : i64)
  ^bb167:  // pred: ^bb156
    %984 = llvm.intr.stacksave : !llvm.ptr
    %985 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %928, %985 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %986 = llvm.insertvalue %985, %596[1] : !llvm.struct<(i64, ptr)> 
    %987 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %928, %987 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %988 = llvm.insertvalue %987, %596[1] : !llvm.struct<(i64, ptr)> 
    %989 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %986, %989 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %990 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %988, %990 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %989, %990) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %984 : !llvm.ptr
    %991 = llvm.intr.stacksave : !llvm.ptr
    %992 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %937, %992 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %993 = llvm.insertvalue %992, %596[1] : !llvm.struct<(i64, ptr)> 
    %994 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %937, %994 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %995 = llvm.insertvalue %994, %596[1] : !llvm.struct<(i64, ptr)> 
    %996 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %993, %996 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %997 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %995, %997 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %996, %997) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %991 : !llvm.ptr
    %998 = llvm.add %909, %35 : i64
    llvm.br ^bb154(%998 : i64)
  ^bb168:  // pred: ^bb154
    %999 = llvm.add %907, %35 : i64
    llvm.br ^bb152(%999 : i64)
  ^bb169:  // pred: ^bb152
    %1000 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %1001 = llvm.ptrtoint %1000 : !llvm.ptr to i64
    %1002 = llvm.add %1001, %57 : i64
    %1003 = llvm.urem %1002, %31  : i64
    %1004 = llvm.sub %1002, %1003 : i64
    %1005 = llvm.inttoptr %1004 : i64 to !llvm.ptr
    %1006 = llvm.insertvalue %1000, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1007 = llvm.insertvalue %1005, %1006[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1008 = llvm.insertvalue %17, %1007[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1009 = llvm.insertvalue %16, %1008[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1010 = llvm.insertvalue %34, %1009[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1011 = llvm.insertvalue %15, %1010[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1012 = llvm.insertvalue %31, %1011[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1013 = llvm.insertvalue %33, %1012[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1014 = llvm.insertvalue %11, %1013[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1015 = llvm.insertvalue %16, %1014[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1016 = llvm.insertvalue %16, %1015[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1017 = llvm.intr.stacksave : !llvm.ptr
    %1018 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %889, %1018 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %1019 = llvm.insertvalue %1018, %596[1] : !llvm.struct<(i64, ptr)> 
    %1020 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %1016, %1020 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %1021 = llvm.insertvalue %1020, %596[1] : !llvm.struct<(i64, ptr)> 
    %1022 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %1019, %1022 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %1023 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %1021, %1023 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %1022, %1023) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %1017 : !llvm.ptr
    %1024 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %1025 = llvm.ptrtoint %1024 : !llvm.ptr to i64
    %1026 = llvm.add %1025, %57 : i64
    %1027 = llvm.urem %1026, %31  : i64
    %1028 = llvm.sub %1026, %1027 : i64
    %1029 = llvm.inttoptr %1028 : i64 to !llvm.ptr
    "llvm.intr.memcpy"(%1029, %1005, %611) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1030 = llvm.insertvalue %1024, %3[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1031 = llvm.insertvalue %1029, %1030[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1032 = llvm.insertvalue %16, %1031[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1033 = llvm.insertvalue %16, %1032[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1034 = llvm.insertvalue %34, %1033[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1035 = llvm.insertvalue %15, %1034[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1036 = llvm.insertvalue %31, %1035[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1037 = llvm.insertvalue %33, %1036[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1038 = llvm.insertvalue %11, %1037[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1039 = llvm.insertvalue %16, %1038[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1040 = llvm.insertvalue %16, %1039[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %1041 = llvm.intr.stacksave : !llvm.ptr
    %1042 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %906, %1042 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %1043 = llvm.insertvalue %1042, %596[1] : !llvm.struct<(i64, ptr)> 
    %1044 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %1040, %1044 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %1045 = llvm.insertvalue %1044, %596[1] : !llvm.struct<(i64, ptr)> 
    %1046 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %1043, %1046 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %1047 = llvm.alloca %16 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %1045, %1047 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%75, %1046, %1047) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %1041 : !llvm.ptr
    %1048 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %1049 = llvm.ptrtoint %1048 : !llvm.ptr to i64
    %1050 = llvm.add %1049, %57 : i64
    %1051 = llvm.urem %1050, %31  : i64
    %1052 = llvm.sub %1050, %1051 : i64
    %1053 = llvm.inttoptr %1052 : i64 to !llvm.ptr
    %1054 = llvm.extractvalue %170[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1055 = llvm.mul %1054, %16 : i64
    %1056 = llvm.extractvalue %170[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1057 = llvm.mul %1055, %1056 : i64
    %1058 = llvm.extractvalue %170[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1059 = llvm.mul %1057, %1058 : i64
    %1060 = llvm.mul %1059, %75 : i64
    %1061 = llvm.extractvalue %170[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1062 = llvm.extractvalue %170[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1063 = llvm.getelementptr %1061[%1062] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1053, %1063, %1060) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1064 = llvm.mul %168, %6 : i64
    %1065 = llvm.mul %146, %34 : i64
    %1066 = llvm.add %1064, %1065 : i64
    %1067 = llvm.mul %165, %16 : i64
    %1068 = llvm.mul %1067, %34 : i64
    %1069 = llvm.mul %1068, %75 : i64
    %1070 = llvm.getelementptr %1053[%1066] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1070, %820, %1069) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1071 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %1072 = llvm.ptrtoint %1071 : !llvm.ptr to i64
    %1073 = llvm.add %1072, %57 : i64
    %1074 = llvm.urem %1073, %31  : i64
    %1075 = llvm.sub %1073, %1074 : i64
    %1076 = llvm.inttoptr %1075 : i64 to !llvm.ptr
    %1077 = llvm.extractvalue %171[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1078 = llvm.mul %1077, %16 : i64
    %1079 = llvm.extractvalue %171[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1080 = llvm.mul %1078, %1079 : i64
    %1081 = llvm.extractvalue %171[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1082 = llvm.mul %1080, %1081 : i64
    %1083 = llvm.mul %1082, %75 : i64
    %1084 = llvm.extractvalue %171[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1085 = llvm.extractvalue %171[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1086 = llvm.getelementptr %1084[%1085] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1076, %1086, %1083) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1087 = llvm.getelementptr %1076[%1066] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1087, %1029, %1069) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1088 = llvm.getelementptr %48[786432] : (!llvm.ptr) -> !llvm.ptr, f32
    %1089 = llvm.ptrtoint %1088 : !llvm.ptr to i64
    %1090 = llvm.add %1089, %31 : i64
    %1091 = llvm.call @malloc(%1090) : (i64) -> !llvm.ptr
    %1092 = llvm.ptrtoint %1091 : !llvm.ptr to i64
    %1093 = llvm.add %1092, %57 : i64
    %1094 = llvm.urem %1093, %31  : i64
    %1095 = llvm.sub %1093, %1094 : i64
    %1096 = llvm.inttoptr %1095 : i64 to !llvm.ptr
    %1097 = llvm.mul %165, %32 : i64
    %1098 = llvm.mul %1097, %34 : i64
    %1099 = llvm.mul %1098, %75 : i64
    %1100 = llvm.getelementptr %1053[%1064] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1096, %1100, %1099) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1101 = llvm.call @malloc(%1090) : (i64) -> !llvm.ptr
    %1102 = llvm.ptrtoint %1101 : !llvm.ptr to i64
    %1103 = llvm.add %1102, %57 : i64
    %1104 = llvm.urem %1103, %31  : i64
    %1105 = llvm.sub %1103, %1104 : i64
    %1106 = llvm.inttoptr %1105 : i64 to !llvm.ptr
    %1107 = llvm.getelementptr %1076[%1064] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1106, %1107, %1099) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1108 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %1109 = llvm.ptrtoint %1108 : !llvm.ptr to i64
    %1110 = llvm.add %1109, %57 : i64
    %1111 = llvm.urem %1110, %31  : i64
    %1112 = llvm.sub %1110, %1111 : i64
    %1113 = llvm.inttoptr %1112 : i64 to !llvm.ptr
    %1114 = llvm.mul %608, %31 : i64
    %1115 = llvm.mul %1114, %75 : i64
    "llvm.intr.memcpy"(%1113, %50, %1115) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb170(%17 : i64)
  ^bb170(%1116: i64):  // 2 preds: ^bb169, ^bb261
    %1117 = llvm.icmp "slt" %1116, %15 : i64
    llvm.cond_br %1117, ^bb171, ^bb262
  ^bb171:  // pred: ^bb170
    %1118 = llvm.mul %1116, %20 : i64
    %1119 = llvm.getelementptr %48[1024] : (!llvm.ptr) -> !llvm.ptr, f32
    %1120 = llvm.ptrtoint %1119 : !llvm.ptr to i64
    %1121 = llvm.add %1120, %31 : i64
    %1122 = llvm.call @malloc(%1121) : (i64) -> !llvm.ptr
    %1123 = llvm.ptrtoint %1122 : !llvm.ptr to i64
    %1124 = llvm.add %1123, %57 : i64
    %1125 = llvm.urem %1124, %31  : i64
    %1126 = llvm.sub %1124, %1125 : i64
    %1127 = llvm.inttoptr %1126 : i64 to !llvm.ptr
    llvm.br ^bb172(%17 : i64)
  ^bb172(%1128: i64):  // 2 preds: ^bb171, ^bb179
    %1129 = llvm.icmp "slt" %1128, %32 : i64
    llvm.cond_br %1129, ^bb173, ^bb180
  ^bb173:  // pred: ^bb172
    llvm.br ^bb174(%17 : i64)
  ^bb174(%1130: i64):  // 2 preds: ^bb173, ^bb178
    %1131 = llvm.icmp "slt" %1130, %16 : i64
    llvm.cond_br %1131, ^bb175, ^bb179
  ^bb175:  // pred: ^bb174
    llvm.br ^bb176(%17 : i64)
  ^bb176(%1132: i64):  // 2 preds: ^bb175, ^bb177
    %1133 = llvm.icmp "slt" %1132, %35 : i64
    llvm.cond_br %1133, ^bb177, ^bb178
  ^bb177:  // pred: ^bb176
    %1134 = llvm.getelementptr %1127[%1128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1135 = llvm.mul %1130, %32 : i64
    %1136 = llvm.add %1135, %1132 : i64
    %1137 = llvm.getelementptr %1134[%1136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1137 : f32, !llvm.ptr
    %1138 = llvm.add %1132, %16 : i64
    llvm.br ^bb176(%1138 : i64)
  ^bb178:  // pred: ^bb176
    %1139 = llvm.add %1130, %16 : i64
    llvm.br ^bb174(%1139 : i64)
  ^bb179:  // pred: ^bb174
    %1140 = llvm.mul %165, %35 : i64
    %1141 = llvm.mul %1140, %75 : i64
    %1142 = llvm.getelementptr %1127[%1128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1142, %1142, %1141) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1143 = llvm.add %1128, %35 : i64
    llvm.br ^bb172(%1143 : i64)
  ^bb180:  // pred: ^bb172
    llvm.br ^bb181(%17 : i64)
  ^bb181(%1144: i64):  // 2 preds: ^bb180, ^bb194
    %1145 = llvm.icmp "slt" %1144, %32 : i64
    llvm.cond_br %1145, ^bb182, ^bb195
  ^bb182:  // pred: ^bb181
    llvm.br ^bb183(%17 : i64)
  ^bb183(%1146: i64):  // 2 preds: ^bb182, ^bb193
    %1147 = llvm.icmp "slt" %1146, %31 : i64
    llvm.cond_br %1147, ^bb184, ^bb194
  ^bb184:  // pred: ^bb183
    %1148 = llvm.add %1118, %1146 : i64
    %1149 = llvm.mul %1144, %34 : i64
    %1150 = llvm.add %1118, %1149 : i64
    %1151 = llvm.add %1150, %1146 : i64
    llvm.br ^bb185(%17 : i64)
  ^bb185(%1152: i64):  // 2 preds: ^bb184, ^bb192
    %1153 = llvm.icmp "slt" %1152, %16 : i64
    llvm.cond_br %1153, ^bb186, ^bb193
  ^bb186:  // pred: ^bb185
    llvm.br ^bb187(%17 : i64)
  ^bb187(%1154: i64):  // 2 preds: ^bb186, ^bb191
    %1155 = llvm.icmp "slt" %1154, %35 : i64
    llvm.cond_br %1155, ^bb188, ^bb192
  ^bb188:  // pred: ^bb187
    llvm.br ^bb189(%17 : i64)
  ^bb189(%1156: i64):  // 2 preds: ^bb188, ^bb190
    %1157 = llvm.icmp "slt" %1156, %35 : i64
    llvm.cond_br %1157, ^bb190, ^bb191
  ^bb190:  // pred: ^bb189
    %1158 = llvm.getelementptr %607[%1148] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1159 = llvm.mul %1152, %34 : i64
    %1160 = llvm.add %1159, %1156 : i64
    %1161 = llvm.getelementptr %1158[%1160] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1162 = llvm.load %1161 : !llvm.ptr -> f32
    %1163 = llvm.getelementptr %1096[%1151] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1164 = llvm.mul %1154, %34 : i64
    %1165 = llvm.add %1164, %1156 : i64
    %1166 = llvm.getelementptr %1163[%1165] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1167 = llvm.load %1166 : !llvm.ptr -> f32
    %1168 = llvm.getelementptr %1127[%1144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1169 = llvm.mul %1152, %32 : i64
    %1170 = llvm.add %1169, %1154 : i64
    %1171 = llvm.getelementptr %1168[%1170] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1172 = llvm.load %1171 : !llvm.ptr -> f32
    %1173 = llvm.add %1154, %1144 : i64
    %1174 = llvm.icmp "sle" %1173, %146 : i64
    %1175 = llvm.select %1174, %18, %19 : i1, f32
    %1176 = llvm.fmul %1162, %1167  : f32
    %1177 = llvm.fadd %1172, %1176  : f32
    %1178 = llvm.fcmp "ugt" %1175, %27 : f32
    %1179 = llvm.select %1178, %1177, %26 : i1, f32
    llvm.store %1179, %1171 : f32, !llvm.ptr
    %1180 = llvm.add %1156, %16 : i64
    llvm.br ^bb189(%1180 : i64)
  ^bb191:  // pred: ^bb189
    %1181 = llvm.add %1154, %16 : i64
    llvm.br ^bb187(%1181 : i64)
  ^bb192:  // pred: ^bb187
    %1182 = llvm.add %1152, %16 : i64
    llvm.br ^bb185(%1182 : i64)
  ^bb193:  // pred: ^bb185
    %1183 = llvm.mul %165, %35 : i64
    %1184 = llvm.mul %1183, %75 : i64
    %1185 = llvm.getelementptr %1127[%1144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1185, %1185, %1184) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1186 = llvm.add %1146, %35 : i64
    llvm.br ^bb183(%1186 : i64)
  ^bb194:  // pred: ^bb183
    %1187 = llvm.add %1144, %35 : i64
    llvm.br ^bb181(%1187 : i64)
  ^bb195:  // pred: ^bb181
    %1188 = llvm.call @malloc(%1121) : (i64) -> !llvm.ptr
    %1189 = llvm.ptrtoint %1188 : !llvm.ptr to i64
    %1190 = llvm.add %1189, %57 : i64
    %1191 = llvm.urem %1190, %31  : i64
    %1192 = llvm.sub %1190, %1191 : i64
    %1193 = llvm.inttoptr %1192 : i64 to !llvm.ptr
    llvm.br ^bb196(%17 : i64)
  ^bb196(%1194: i64):  // 2 preds: ^bb195, ^bb203
    %1195 = llvm.icmp "slt" %1194, %32 : i64
    llvm.cond_br %1195, ^bb197, ^bb204
  ^bb197:  // pred: ^bb196
    llvm.br ^bb198(%17 : i64)
  ^bb198(%1196: i64):  // 2 preds: ^bb197, ^bb202
    %1197 = llvm.icmp "slt" %1196, %16 : i64
    llvm.cond_br %1197, ^bb199, ^bb203
  ^bb199:  // pred: ^bb198
    llvm.br ^bb200(%17 : i64)
  ^bb200(%1198: i64):  // 2 preds: ^bb199, ^bb201
    %1199 = llvm.icmp "slt" %1198, %35 : i64
    llvm.cond_br %1199, ^bb201, ^bb202
  ^bb201:  // pred: ^bb200
    %1200 = llvm.getelementptr %1127[%1194] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1201 = llvm.mul %1196, %32 : i64
    %1202 = llvm.add %1201, %1198 : i64
    %1203 = llvm.getelementptr %1200[%1202] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1204 = llvm.load %1203 : !llvm.ptr -> f32
    %1205 = llvm.fmul %1204, %21  : f32
    %1206 = llvm.getelementptr %1193[%1194] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1207 = llvm.getelementptr %1206[%1202] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1205, %1207 : f32, !llvm.ptr
    %1208 = llvm.add %1198, %16 : i64
    llvm.br ^bb200(%1208 : i64)
  ^bb202:  // pred: ^bb200
    %1209 = llvm.add %1196, %16 : i64
    llvm.br ^bb198(%1209 : i64)
  ^bb203:  // pred: ^bb198
    %1210 = llvm.mul %165, %35 : i64
    %1211 = llvm.mul %1210, %75 : i64
    %1212 = llvm.getelementptr %1193[%1194] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1212, %1212, %1211) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1213 = llvm.add %1194, %35 : i64
    llvm.br ^bb196(%1213 : i64)
  ^bb204:  // pred: ^bb196
    %1214 = llvm.call @malloc(%173) : (i64) -> !llvm.ptr
    %1215 = llvm.ptrtoint %1214 : !llvm.ptr to i64
    %1216 = llvm.add %1215, %57 : i64
    %1217 = llvm.urem %1216, %31  : i64
    %1218 = llvm.sub %1216, %1217 : i64
    %1219 = llvm.inttoptr %1218 : i64 to !llvm.ptr
    llvm.br ^bb205(%17 : i64)
  ^bb205(%1220: i64):  // 2 preds: ^bb204, ^bb206
    %1221 = llvm.icmp "slt" %1220, %16 : i64
    llvm.cond_br %1221, ^bb206, ^bb207
  ^bb206:  // pred: ^bb205
    %1222 = llvm.getelementptr %1219[%1220] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %43, %1222 : f32, !llvm.ptr
    %1223 = llvm.add %1220, %16 : i64
    llvm.br ^bb205(%1223 : i64)
  ^bb207:  // pred: ^bb205
    llvm.br ^bb208(%17 : i64)
  ^bb208(%1224: i64):  // 2 preds: ^bb207, ^bb215
    %1225 = llvm.icmp "slt" %1224, %32 : i64
    llvm.cond_br %1225, ^bb209, ^bb216
  ^bb209:  // pred: ^bb208
    llvm.br ^bb210(%17 : i64)
  ^bb210(%1226: i64):  // 2 preds: ^bb209, ^bb214
    %1227 = llvm.icmp "slt" %1226, %16 : i64
    llvm.cond_br %1227, ^bb211, ^bb215
  ^bb211:  // pred: ^bb210
    llvm.br ^bb212(%17 : i64)
  ^bb212(%1228: i64):  // 2 preds: ^bb211, ^bb213
    %1229 = llvm.icmp "slt" %1228, %35 : i64
    llvm.cond_br %1229, ^bb213, ^bb214
  ^bb213:  // pred: ^bb212
    %1230 = llvm.getelementptr %1193[%1224] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1231 = llvm.mul %1226, %32 : i64
    %1232 = llvm.add %1231, %1228 : i64
    %1233 = llvm.getelementptr %1230[%1232] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1234 = llvm.load %1233 : !llvm.ptr -> f32
    %1235 = llvm.getelementptr %1219[%1226] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1236 = llvm.load %1235 : !llvm.ptr -> f32
    %1237 = llvm.intr.maxnum(%1234, %1236)  : (f32, f32) -> f32
    llvm.store %1237, %1235 : f32, !llvm.ptr
    %1238 = llvm.add %1228, %16 : i64
    llvm.br ^bb212(%1238 : i64)
  ^bb214:  // pred: ^bb212
    %1239 = llvm.add %1226, %16 : i64
    llvm.br ^bb210(%1239 : i64)
  ^bb215:  // pred: ^bb210
    %1240 = llvm.add %1224, %35 : i64
    llvm.br ^bb208(%1240 : i64)
  ^bb216:  // pred: ^bb208
    %1241 = llvm.call @malloc(%1121) : (i64) -> !llvm.ptr
    %1242 = llvm.ptrtoint %1241 : !llvm.ptr to i64
    %1243 = llvm.add %1242, %57 : i64
    %1244 = llvm.urem %1243, %31  : i64
    %1245 = llvm.sub %1243, %1244 : i64
    %1246 = llvm.inttoptr %1245 : i64 to !llvm.ptr
    llvm.br ^bb217(%17 : i64)
  ^bb217(%1247: i64):  // 2 preds: ^bb216, ^bb224
    %1248 = llvm.icmp "slt" %1247, %32 : i64
    llvm.cond_br %1248, ^bb218, ^bb225
  ^bb218:  // pred: ^bb217
    llvm.br ^bb219(%17 : i64)
  ^bb219(%1249: i64):  // 2 preds: ^bb218, ^bb223
    %1250 = llvm.icmp "slt" %1249, %16 : i64
    llvm.cond_br %1250, ^bb220, ^bb224
  ^bb220:  // pred: ^bb219
    llvm.br ^bb221(%17 : i64)
  ^bb221(%1251: i64):  // 2 preds: ^bb220, ^bb222
    %1252 = llvm.icmp "slt" %1251, %35 : i64
    llvm.cond_br %1252, ^bb222, ^bb223
  ^bb222:  // pred: ^bb221
    %1253 = llvm.getelementptr %1193[%1247] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1254 = llvm.mul %1249, %32 : i64
    %1255 = llvm.add %1254, %1251 : i64
    %1256 = llvm.getelementptr %1253[%1255] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1257 = llvm.load %1256 : !llvm.ptr -> f32
    %1258 = llvm.getelementptr %1219[%1249] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1259 = llvm.load %1258 : !llvm.ptr -> f32
    %1260 = llvm.fsub %1257, %1259  : f32
    %1261 = llvm.intr.exp(%1260)  : (f32) -> f32
    %1262 = llvm.getelementptr %1246[%1247] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1263 = llvm.getelementptr %1262[%1255] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1261, %1263 : f32, !llvm.ptr
    %1264 = llvm.add %1251, %16 : i64
    llvm.br ^bb221(%1264 : i64)
  ^bb223:  // pred: ^bb221
    %1265 = llvm.add %1249, %16 : i64
    llvm.br ^bb219(%1265 : i64)
  ^bb224:  // pred: ^bb219
    %1266 = llvm.mul %165, %35 : i64
    %1267 = llvm.mul %1266, %75 : i64
    %1268 = llvm.getelementptr %1246[%1247] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1268, %1268, %1267) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1269 = llvm.add %1247, %35 : i64
    llvm.br ^bb217(%1269 : i64)
  ^bb225:  // pred: ^bb217
    %1270 = llvm.call @malloc(%173) : (i64) -> !llvm.ptr
    %1271 = llvm.ptrtoint %1270 : !llvm.ptr to i64
    %1272 = llvm.add %1271, %57 : i64
    %1273 = llvm.urem %1272, %31  : i64
    %1274 = llvm.sub %1272, %1273 : i64
    %1275 = llvm.inttoptr %1274 : i64 to !llvm.ptr
    llvm.br ^bb226(%17 : i64)
  ^bb226(%1276: i64):  // 2 preds: ^bb225, ^bb227
    %1277 = llvm.icmp "slt" %1276, %16 : i64
    llvm.cond_br %1277, ^bb227, ^bb228
  ^bb227:  // pred: ^bb226
    %1278 = llvm.getelementptr %1275[%1276] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1278 : f32, !llvm.ptr
    %1279 = llvm.add %1276, %16 : i64
    llvm.br ^bb226(%1279 : i64)
  ^bb228:  // pred: ^bb226
    llvm.br ^bb229(%17 : i64)
  ^bb229(%1280: i64):  // 2 preds: ^bb228, ^bb236
    %1281 = llvm.icmp "slt" %1280, %32 : i64
    llvm.cond_br %1281, ^bb230, ^bb237
  ^bb230:  // pred: ^bb229
    llvm.br ^bb231(%17 : i64)
  ^bb231(%1282: i64):  // 2 preds: ^bb230, ^bb235
    %1283 = llvm.icmp "slt" %1282, %16 : i64
    llvm.cond_br %1283, ^bb232, ^bb236
  ^bb232:  // pred: ^bb231
    llvm.br ^bb233(%17 : i64)
  ^bb233(%1284: i64):  // 2 preds: ^bb232, ^bb234
    %1285 = llvm.icmp "slt" %1284, %35 : i64
    llvm.cond_br %1285, ^bb234, ^bb235
  ^bb234:  // pred: ^bb233
    %1286 = llvm.getelementptr %1246[%1280] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1287 = llvm.mul %1282, %32 : i64
    %1288 = llvm.add %1287, %1284 : i64
    %1289 = llvm.getelementptr %1286[%1288] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1290 = llvm.load %1289 : !llvm.ptr -> f32
    %1291 = llvm.getelementptr %1275[%1282] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1292 = llvm.load %1291 : !llvm.ptr -> f32
    %1293 = llvm.fadd %1290, %1292  : f32
    llvm.store %1293, %1291 : f32, !llvm.ptr
    %1294 = llvm.add %1284, %16 : i64
    llvm.br ^bb233(%1294 : i64)
  ^bb235:  // pred: ^bb233
    %1295 = llvm.add %1282, %16 : i64
    llvm.br ^bb231(%1295 : i64)
  ^bb236:  // pred: ^bb231
    %1296 = llvm.add %1280, %35 : i64
    llvm.br ^bb229(%1296 : i64)
  ^bb237:  // pred: ^bb229
    %1297 = llvm.getelementptr %48[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %1298 = llvm.ptrtoint %1297 : !llvm.ptr to i64
    %1299 = llvm.add %1298, %31 : i64
    %1300 = llvm.call @malloc(%1299) : (i64) -> !llvm.ptr
    %1301 = llvm.ptrtoint %1300 : !llvm.ptr to i64
    %1302 = llvm.add %1301, %57 : i64
    %1303 = llvm.urem %1302, %31  : i64
    %1304 = llvm.sub %1302, %1303 : i64
    %1305 = llvm.inttoptr %1304 : i64 to !llvm.ptr
    llvm.br ^bb238(%17 : i64)
  ^bb238(%1306: i64):  // 2 preds: ^bb237, ^bb245
    %1307 = llvm.icmp "slt" %1306, %31 : i64
    llvm.cond_br %1307, ^bb239, ^bb246
  ^bb239:  // pred: ^bb238
    llvm.br ^bb240(%17 : i64)
  ^bb240(%1308: i64):  // 2 preds: ^bb239, ^bb244
    %1309 = llvm.icmp "slt" %1308, %16 : i64
    llvm.cond_br %1309, ^bb241, ^bb245
  ^bb241:  // pred: ^bb240
    llvm.br ^bb242(%17 : i64)
  ^bb242(%1310: i64):  // 2 preds: ^bb241, ^bb243
    %1311 = llvm.icmp "slt" %1310, %35 : i64
    llvm.cond_br %1311, ^bb243, ^bb244
  ^bb243:  // pred: ^bb242
    %1312 = llvm.getelementptr %1305[%1306] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1313 = llvm.mul %1308, %31 : i64
    %1314 = llvm.add %1313, %1310 : i64
    %1315 = llvm.getelementptr %1312[%1314] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1315 : f32, !llvm.ptr
    %1316 = llvm.add %1310, %16 : i64
    llvm.br ^bb242(%1316 : i64)
  ^bb244:  // pred: ^bb242
    %1317 = llvm.add %1308, %16 : i64
    llvm.br ^bb240(%1317 : i64)
  ^bb245:  // pred: ^bb240
    %1318 = llvm.mul %165, %35 : i64
    %1319 = llvm.mul %1318, %75 : i64
    %1320 = llvm.getelementptr %1305[%1306] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1320, %1320, %1319) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1321 = llvm.add %1306, %35 : i64
    llvm.br ^bb238(%1321 : i64)
  ^bb246:  // pred: ^bb238
    %1322 = llvm.call @malloc(%1299) : (i64) -> !llvm.ptr
    %1323 = llvm.ptrtoint %1322 : !llvm.ptr to i64
    %1324 = llvm.add %1323, %57 : i64
    %1325 = llvm.urem %1324, %31  : i64
    %1326 = llvm.sub %1324, %1325 : i64
    %1327 = llvm.inttoptr %1326 : i64 to !llvm.ptr
    %1328 = llvm.mul %165, %31 : i64
    %1329 = llvm.mul %1328, %75 : i64
    "llvm.intr.memcpy"(%1327, %1305, %1329) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb247(%17 : i64)
  ^bb247(%1330: i64):  // 2 preds: ^bb246, ^bb260
    %1331 = llvm.icmp "slt" %1330, %31 : i64
    llvm.cond_br %1331, ^bb248, ^bb261
  ^bb248:  // pred: ^bb247
    llvm.br ^bb249(%17 : i64)
  ^bb249(%1332: i64):  // 2 preds: ^bb248, ^bb259
    %1333 = llvm.icmp "slt" %1332, %32 : i64
    llvm.cond_br %1333, ^bb250, ^bb260
  ^bb250:  // pred: ^bb249
    %1334 = llvm.mul %1332, %34 : i64
    %1335 = llvm.add %1118, %1334 : i64
    %1336 = llvm.add %1335, %1330 : i64
    llvm.br ^bb251(%17 : i64)
  ^bb251(%1337: i64):  // 2 preds: ^bb250, ^bb258
    %1338 = llvm.icmp "slt" %1337, %16 : i64
    llvm.cond_br %1338, ^bb252, ^bb259
  ^bb252:  // pred: ^bb251
    llvm.br ^bb253(%17 : i64)
  ^bb253(%1339: i64):  // 2 preds: ^bb252, ^bb257
    %1340 = llvm.icmp "slt" %1339, %35 : i64
    llvm.cond_br %1340, ^bb254, ^bb258
  ^bb254:  // pred: ^bb253
    llvm.br ^bb255(%17 : i64)
  ^bb255(%1341: i64):  // 2 preds: ^bb254, ^bb256
    %1342 = llvm.icmp "slt" %1341, %35 : i64
    llvm.cond_br %1342, ^bb256, ^bb257
  ^bb256:  // pred: ^bb255
    %1343 = llvm.getelementptr %1246[%1332] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1344 = llvm.mul %1337, %32 : i64
    %1345 = llvm.add %1344, %1341 : i64
    %1346 = llvm.getelementptr %1343[%1345] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1347 = llvm.load %1346 : !llvm.ptr -> f32
    %1348 = llvm.getelementptr %1275[%1337] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1349 = llvm.load %1348 : !llvm.ptr -> f32
    %1350 = llvm.getelementptr %1106[%1336] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1351 = llvm.mul %1341, %34 : i64
    %1352 = llvm.add %1351, %1339 : i64
    %1353 = llvm.getelementptr %1350[%1352] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1354 = llvm.load %1353 : !llvm.ptr -> f32
    %1355 = llvm.getelementptr %1327[%1330] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1356 = llvm.mul %1337, %31 : i64
    %1357 = llvm.add %1356, %1339 : i64
    %1358 = llvm.getelementptr %1355[%1357] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1359 = llvm.load %1358 : !llvm.ptr -> f32
    %1360 = llvm.fdiv %1347, %1349  : f32
    %1361 = llvm.fmul %1360, %1354  : f32
    %1362 = llvm.fadd %1359, %1361  : f32
    llvm.store %1362, %1358 : f32, !llvm.ptr
    %1363 = llvm.add %1341, %16 : i64
    llvm.br ^bb255(%1363 : i64)
  ^bb257:  // pred: ^bb255
    %1364 = llvm.add %1339, %16 : i64
    llvm.br ^bb253(%1364 : i64)
  ^bb258:  // pred: ^bb253
    %1365 = llvm.add %1337, %16 : i64
    llvm.br ^bb251(%1365 : i64)
  ^bb259:  // pred: ^bb251
    %1366 = llvm.mul %165, %35 : i64
    %1367 = llvm.mul %1366, %75 : i64
    %1368 = llvm.getelementptr %1327[%1330] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1368, %1368, %1367) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1369 = llvm.add %1332, %35 : i64
    llvm.br ^bb249(%1369 : i64)
  ^bb260:  // pred: ^bb249
    %1370 = llvm.add %1330, %35 : i64
    llvm.br ^bb247(%1370 : i64)
  ^bb261:  // pred: ^bb247
    %1371 = llvm.mul %1116, %31 : i64
    %1372 = llvm.mul %1067, %31 : i64
    %1373 = llvm.mul %1372, %75 : i64
    %1374 = llvm.getelementptr %1113[%1371] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1374, %1327, %1373) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1375 = llvm.add %1116, %16 : i64
    llvm.br ^bb170(%1375 : i64)
  ^bb262:  // pred: ^bb170
    %1376 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %1377 = llvm.ptrtoint %1376 : !llvm.ptr to i64
    %1378 = llvm.add %1377, %57 : i64
    %1379 = llvm.urem %1378, %31  : i64
    %1380 = llvm.sub %1378, %1379 : i64
    %1381 = llvm.inttoptr %1380 : i64 to !llvm.ptr
    llvm.br ^bb263(%17 : i64)
  ^bb263(%1382: i64):  // 2 preds: ^bb262, ^bb270
    %1383 = llvm.icmp "slt" %1382, %34 : i64
    llvm.cond_br %1383, ^bb264, ^bb271
  ^bb264:  // pred: ^bb263
    llvm.br ^bb265(%17 : i64)
  ^bb265(%1384: i64):  // 2 preds: ^bb264, ^bb269
    %1385 = llvm.icmp "slt" %1384, %16 : i64
    llvm.cond_br %1385, ^bb266, ^bb270
  ^bb266:  // pred: ^bb265
    llvm.br ^bb267(%17 : i64)
  ^bb267(%1386: i64):  // 2 preds: ^bb266, ^bb268
    %1387 = llvm.icmp "slt" %1386, %35 : i64
    llvm.cond_br %1387, ^bb268, ^bb269
  ^bb268:  // pred: ^bb267
    %1388 = llvm.getelementptr %1381[%1382] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1389 = llvm.mul %1384, %34 : i64
    %1390 = llvm.add %1389, %1386 : i64
    %1391 = llvm.getelementptr %1388[%1390] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1391 : f32, !llvm.ptr
    %1392 = llvm.add %1386, %16 : i64
    llvm.br ^bb267(%1392 : i64)
  ^bb269:  // pred: ^bb267
    %1393 = llvm.add %1384, %16 : i64
    llvm.br ^bb265(%1393 : i64)
  ^bb270:  // pred: ^bb265
    %1394 = llvm.mul %165, %35 : i64
    %1395 = llvm.mul %1394, %75 : i64
    %1396 = llvm.getelementptr %1381[%1382] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1396, %1396, %1395) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1397 = llvm.add %1382, %35 : i64
    llvm.br ^bb263(%1397 : i64)
  ^bb271:  // pred: ^bb263
    llvm.br ^bb272(%17 : i64)
  ^bb272(%1398: i64):  // 2 preds: ^bb271, ^bb285
    %1399 = llvm.icmp "slt" %1398, %34 : i64
    llvm.cond_br %1399, ^bb273, ^bb286
  ^bb273:  // pred: ^bb272
    llvm.br ^bb274(%17 : i64)
  ^bb274(%1400: i64):  // 2 preds: ^bb273, ^bb284
    %1401 = llvm.icmp "slt" %1400, %34 : i64
    llvm.cond_br %1401, ^bb275, ^bb285
  ^bb275:  // pred: ^bb274
    llvm.br ^bb276(%17 : i64)
  ^bb276(%1402: i64):  // 2 preds: ^bb275, ^bb283
    %1403 = llvm.icmp "slt" %1402, %16 : i64
    llvm.cond_br %1403, ^bb277, ^bb284
  ^bb277:  // pred: ^bb276
    llvm.br ^bb278(%17 : i64)
  ^bb278(%1404: i64):  // 2 preds: ^bb277, ^bb282
    %1405 = llvm.icmp "slt" %1404, %35 : i64
    llvm.cond_br %1405, ^bb279, ^bb283
  ^bb279:  // pred: ^bb278
    llvm.br ^bb280(%17 : i64)
  ^bb280(%1406: i64):  // 2 preds: ^bb279, ^bb281
    %1407 = llvm.icmp "slt" %1406, %35 : i64
    llvm.cond_br %1407, ^bb281, ^bb282
  ^bb281:  // pred: ^bb280
    %1408 = llvm.getelementptr %1113[%1400] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1409 = llvm.mul %1402, %34 : i64
    %1410 = llvm.add %1409, %1406 : i64
    %1411 = llvm.getelementptr %1408[%1410] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1412 = llvm.load %1411 : !llvm.ptr -> f32
    %1413 = llvm.getelementptr %1381[%1398] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1414 = llvm.add %1409, %1404 : i64
    %1415 = llvm.getelementptr %1413[%1414] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1416 = llvm.load %1415 : !llvm.ptr -> f32
    %1417 = llvm.fmul %1412, %42  : f32
    %1418 = llvm.fadd %1416, %1417  : f32
    llvm.store %1418, %1415 : f32, !llvm.ptr
    %1419 = llvm.add %1406, %16 : i64
    llvm.br ^bb280(%1419 : i64)
  ^bb282:  // pred: ^bb280
    %1420 = llvm.add %1404, %16 : i64
    llvm.br ^bb278(%1420 : i64)
  ^bb283:  // pred: ^bb278
    %1421 = llvm.add %1402, %16 : i64
    llvm.br ^bb276(%1421 : i64)
  ^bb284:  // pred: ^bb276
    %1422 = llvm.mul %165, %35 : i64
    %1423 = llvm.mul %1422, %75 : i64
    %1424 = llvm.getelementptr %1381[%1398] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1424, %1424, %1423) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1425 = llvm.add %1400, %35 : i64
    llvm.br ^bb274(%1425 : i64)
  ^bb285:  // pred: ^bb274
    %1426 = llvm.add %1398, %35 : i64
    llvm.br ^bb272(%1426 : i64)
  ^bb286:  // pred: ^bb272
    %1427 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %1428 = llvm.ptrtoint %1427 : !llvm.ptr to i64
    %1429 = llvm.add %1428, %57 : i64
    %1430 = llvm.urem %1429, %31  : i64
    %1431 = llvm.sub %1429, %1430 : i64
    %1432 = llvm.inttoptr %1431 : i64 to !llvm.ptr
    llvm.br ^bb287(%17 : i64)
  ^bb287(%1433: i64):  // 2 preds: ^bb286, ^bb294
    %1434 = llvm.icmp "slt" %1433, %34 : i64
    llvm.cond_br %1434, ^bb288, ^bb295
  ^bb288:  // pred: ^bb287
    %1435 = llvm.extractvalue %169[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb289(%17 : i64)
  ^bb289(%1436: i64):  // 2 preds: ^bb288, ^bb293
    %1437 = llvm.icmp "slt" %1436, %16 : i64
    llvm.cond_br %1437, ^bb290, ^bb294
  ^bb290:  // pred: ^bb289
    llvm.br ^bb291(%17 : i64)
  ^bb291(%1438: i64):  // 2 preds: ^bb290, ^bb292
    %1439 = llvm.icmp "slt" %1438, %35 : i64
    llvm.cond_br %1439, ^bb292, ^bb293
  ^bb292:  // pred: ^bb291
    %1440 = llvm.getelementptr %1435[%1433] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1441 = llvm.mul %1436, %34 : i64
    %1442 = llvm.add %1441, %1438 : i64
    %1443 = llvm.getelementptr %1440[%1442] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1444 = llvm.load %1443 : !llvm.ptr -> f32
    %1445 = llvm.getelementptr %1381[%1433] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1446 = llvm.getelementptr %1445[%1442] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1447 = llvm.load %1446 : !llvm.ptr -> f32
    %1448 = llvm.fadd %1444, %1447  : f32
    %1449 = llvm.getelementptr %1432[%1433] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1450 = llvm.getelementptr %1449[%1442] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1448, %1450 : f32, !llvm.ptr
    %1451 = llvm.add %1438, %16 : i64
    llvm.br ^bb291(%1451 : i64)
  ^bb293:  // pred: ^bb291
    %1452 = llvm.add %1436, %16 : i64
    llvm.br ^bb289(%1452 : i64)
  ^bb294:  // pred: ^bb289
    %1453 = llvm.mul %165, %35 : i64
    %1454 = llvm.mul %1453, %75 : i64
    %1455 = llvm.getelementptr %1432[%1433] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1455, %1455, %1454) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1456 = llvm.add %1433, %35 : i64
    llvm.br ^bb287(%1456 : i64)
  ^bb295:  // pred: ^bb287
    %1457 = llvm.call @malloc(%173) : (i64) -> !llvm.ptr
    %1458 = llvm.ptrtoint %1457 : !llvm.ptr to i64
    %1459 = llvm.add %1458, %57 : i64
    %1460 = llvm.urem %1459, %31  : i64
    %1461 = llvm.sub %1459, %1460 : i64
    %1462 = llvm.inttoptr %1461 : i64 to !llvm.ptr
    llvm.br ^bb296(%17 : i64)
  ^bb296(%1463: i64):  // 2 preds: ^bb295, ^bb297
    %1464 = llvm.icmp "slt" %1463, %16 : i64
    llvm.cond_br %1464, ^bb297, ^bb298
  ^bb297:  // pred: ^bb296
    %1465 = llvm.getelementptr %1462[%1463] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1465 : f32, !llvm.ptr
    %1466 = llvm.add %1463, %16 : i64
    llvm.br ^bb296(%1466 : i64)
  ^bb298:  // pred: ^bb296
    llvm.br ^bb299(%17 : i64)
  ^bb299(%1467: i64):  // 2 preds: ^bb298, ^bb306
    %1468 = llvm.icmp "slt" %1467, %34 : i64
    llvm.cond_br %1468, ^bb300, ^bb307
  ^bb300:  // pred: ^bb299
    llvm.br ^bb301(%17 : i64)
  ^bb301(%1469: i64):  // 2 preds: ^bb300, ^bb305
    %1470 = llvm.icmp "slt" %1469, %16 : i64
    llvm.cond_br %1470, ^bb302, ^bb306
  ^bb302:  // pred: ^bb301
    llvm.br ^bb303(%17 : i64)
  ^bb303(%1471: i64):  // 2 preds: ^bb302, ^bb304
    %1472 = llvm.icmp "slt" %1471, %35 : i64
    llvm.cond_br %1472, ^bb304, ^bb305
  ^bb304:  // pred: ^bb303
    %1473 = llvm.getelementptr %1432[%1467] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1474 = llvm.mul %1469, %34 : i64
    %1475 = llvm.add %1474, %1471 : i64
    %1476 = llvm.getelementptr %1473[%1475] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1477 = llvm.load %1476 : !llvm.ptr -> f32
    %1478 = llvm.getelementptr %1462[%1469] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1479 = llvm.load %1478 : !llvm.ptr -> f32
    %1480 = llvm.fmul %1477, %1477  : f32
    %1481 = llvm.fadd %1479, %1480  : f32
    llvm.store %1481, %1478 : f32, !llvm.ptr
    %1482 = llvm.add %1471, %16 : i64
    llvm.br ^bb303(%1482 : i64)
  ^bb305:  // pred: ^bb303
    %1483 = llvm.add %1469, %16 : i64
    llvm.br ^bb301(%1483 : i64)
  ^bb306:  // pred: ^bb301
    %1484 = llvm.add %1467, %35 : i64
    llvm.br ^bb299(%1484 : i64)
  ^bb307:  // pred: ^bb299
    %1485 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %1486 = llvm.ptrtoint %1485 : !llvm.ptr to i64
    %1487 = llvm.add %1486, %57 : i64
    %1488 = llvm.urem %1487, %31  : i64
    %1489 = llvm.sub %1487, %1488 : i64
    %1490 = llvm.inttoptr %1489 : i64 to !llvm.ptr
    llvm.br ^bb308(%17 : i64)
  ^bb308(%1491: i64):  // 2 preds: ^bb307, ^bb315
    %1492 = llvm.icmp "slt" %1491, %34 : i64
    llvm.cond_br %1492, ^bb309, ^bb316
  ^bb309:  // pred: ^bb308
    llvm.br ^bb310(%17 : i64)
  ^bb310(%1493: i64):  // 2 preds: ^bb309, ^bb314
    %1494 = llvm.icmp "slt" %1493, %16 : i64
    llvm.cond_br %1494, ^bb311, ^bb315
  ^bb311:  // pred: ^bb310
    llvm.br ^bb312(%17 : i64)
  ^bb312(%1495: i64):  // 2 preds: ^bb311, ^bb313
    %1496 = llvm.icmp "slt" %1495, %35 : i64
    llvm.cond_br %1496, ^bb313, ^bb314
  ^bb313:  // pred: ^bb312
    %1497 = llvm.getelementptr %1432[%1491] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1498 = llvm.mul %1493, %34 : i64
    %1499 = llvm.add %1498, %1495 : i64
    %1500 = llvm.getelementptr %1497[%1499] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1501 = llvm.load %1500 : !llvm.ptr -> f32
    %1502 = llvm.getelementptr %1462[%1493] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1503 = llvm.load %1502 : !llvm.ptr -> f32
    %1504 = llvm.fdiv %1503, %28  : f32
    %1505 = llvm.fadd %1504, %22  : f32
    %1506 = llvm.intr.sqrt(%1505)  : (f32) -> f32
    %1507 = llvm.fdiv %18, %1506  : f32
    %1508 = llvm.fmul %1501, %1507  : f32
    %1509 = llvm.fmul %1508, %41  : f32
    %1510 = llvm.getelementptr %1490[%1491] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1511 = llvm.getelementptr %1510[%1499] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1509, %1511 : f32, !llvm.ptr
    %1512 = llvm.add %1495, %16 : i64
    llvm.br ^bb312(%1512 : i64)
  ^bb314:  // pred: ^bb312
    %1513 = llvm.add %1493, %16 : i64
    llvm.br ^bb310(%1513 : i64)
  ^bb315:  // pred: ^bb310
    %1514 = llvm.mul %165, %35 : i64
    %1515 = llvm.mul %1514, %75 : i64
    %1516 = llvm.getelementptr %1490[%1491] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1516, %1516, %1515) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1517 = llvm.add %1491, %35 : i64
    llvm.br ^bb308(%1517 : i64)
  ^bb316:  // pred: ^bb308
    %1518 = llvm.getelementptr %48[2048] : (!llvm.ptr) -> !llvm.ptr, f32
    %1519 = llvm.ptrtoint %1518 : !llvm.ptr to i64
    %1520 = llvm.add %1519, %31 : i64
    %1521 = llvm.call @malloc(%1520) : (i64) -> !llvm.ptr
    %1522 = llvm.ptrtoint %1521 : !llvm.ptr to i64
    %1523 = llvm.add %1522, %57 : i64
    %1524 = llvm.urem %1523, %31  : i64
    %1525 = llvm.sub %1523, %1524 : i64
    %1526 = llvm.inttoptr %1525 : i64 to !llvm.ptr
    llvm.br ^bb317(%17 : i64)
  ^bb317(%1527: i64):  // 2 preds: ^bb316, ^bb324
    %1528 = llvm.icmp "slt" %1527, %30 : i64
    llvm.cond_br %1528, ^bb318, ^bb325
  ^bb318:  // pred: ^bb317
    llvm.br ^bb319(%17 : i64)
  ^bb319(%1529: i64):  // 2 preds: ^bb318, ^bb323
    %1530 = llvm.icmp "slt" %1529, %16 : i64
    llvm.cond_br %1530, ^bb320, ^bb324
  ^bb320:  // pred: ^bb319
    llvm.br ^bb321(%17 : i64)
  ^bb321(%1531: i64):  // 2 preds: ^bb320, ^bb322
    %1532 = llvm.icmp "slt" %1531, %35 : i64
    llvm.cond_br %1532, ^bb322, ^bb323
  ^bb322:  // pred: ^bb321
    %1533 = llvm.getelementptr %1526[%1527] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1534 = llvm.mul %1529, %30 : i64
    %1535 = llvm.add %1534, %1531 : i64
    %1536 = llvm.getelementptr %1533[%1535] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1536 : f32, !llvm.ptr
    %1537 = llvm.add %1531, %16 : i64
    llvm.br ^bb321(%1537 : i64)
  ^bb323:  // pred: ^bb321
    %1538 = llvm.add %1529, %16 : i64
    llvm.br ^bb319(%1538 : i64)
  ^bb324:  // pred: ^bb319
    %1539 = llvm.mul %165, %35 : i64
    %1540 = llvm.mul %1539, %75 : i64
    %1541 = llvm.getelementptr %1526[%1527] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1541, %1541, %1540) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1542 = llvm.add %1527, %35 : i64
    llvm.br ^bb317(%1542 : i64)
  ^bb325:  // pred: ^bb317
    llvm.br ^bb326(%17 : i64)
  ^bb326(%1543: i64):  // 2 preds: ^bb325, ^bb339
    %1544 = llvm.icmp "slt" %1543, %30 : i64
    llvm.cond_br %1544, ^bb327, ^bb340
  ^bb327:  // pred: ^bb326
    llvm.br ^bb328(%17 : i64)
  ^bb328(%1545: i64):  // 2 preds: ^bb327, ^bb338
    %1546 = llvm.icmp "slt" %1545, %34 : i64
    llvm.cond_br %1546, ^bb329, ^bb339
  ^bb329:  // pred: ^bb328
    llvm.br ^bb330(%17 : i64)
  ^bb330(%1547: i64):  // 2 preds: ^bb329, ^bb337
    %1548 = llvm.icmp "slt" %1547, %16 : i64
    llvm.cond_br %1548, ^bb331, ^bb338
  ^bb331:  // pred: ^bb330
    llvm.br ^bb332(%17 : i64)
  ^bb332(%1549: i64):  // 2 preds: ^bb331, ^bb336
    %1550 = llvm.icmp "slt" %1549, %35 : i64
    llvm.cond_br %1550, ^bb333, ^bb337
  ^bb333:  // pred: ^bb332
    llvm.br ^bb334(%17 : i64)
  ^bb334(%1551: i64):  // 2 preds: ^bb333, ^bb335
    %1552 = llvm.icmp "slt" %1551, %35 : i64
    llvm.cond_br %1552, ^bb335, ^bb336
  ^bb335:  // pred: ^bb334
    %1553 = llvm.getelementptr %1490[%1545] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1554 = llvm.mul %1547, %34 : i64
    %1555 = llvm.add %1554, %1551 : i64
    %1556 = llvm.getelementptr %1553[%1555] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1557 = llvm.load %1556 : !llvm.ptr -> f32
    %1558 = llvm.getelementptr %1526[%1543] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1559 = llvm.mul %1547, %30 : i64
    %1560 = llvm.add %1559, %1549 : i64
    %1561 = llvm.getelementptr %1558[%1560] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1562 = llvm.load %1561 : !llvm.ptr -> f32
    %1563 = llvm.fmul %1557, %40  : f32
    %1564 = llvm.fadd %1562, %1563  : f32
    llvm.store %1564, %1561 : f32, !llvm.ptr
    %1565 = llvm.add %1551, %16 : i64
    llvm.br ^bb334(%1565 : i64)
  ^bb336:  // pred: ^bb334
    %1566 = llvm.add %1549, %16 : i64
    llvm.br ^bb332(%1566 : i64)
  ^bb337:  // pred: ^bb332
    %1567 = llvm.add %1547, %16 : i64
    llvm.br ^bb330(%1567 : i64)
  ^bb338:  // pred: ^bb330
    %1568 = llvm.mul %165, %35 : i64
    %1569 = llvm.mul %1568, %75 : i64
    %1570 = llvm.getelementptr %1526[%1543] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1570, %1570, %1569) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1571 = llvm.add %1545, %35 : i64
    llvm.br ^bb328(%1571 : i64)
  ^bb339:  // pred: ^bb328
    %1572 = llvm.add %1543, %35 : i64
    llvm.br ^bb326(%1572 : i64)
  ^bb340:  // pred: ^bb326
    %1573 = llvm.call @malloc(%1520) : (i64) -> !llvm.ptr
    %1574 = llvm.ptrtoint %1573 : !llvm.ptr to i64
    %1575 = llvm.add %1574, %57 : i64
    %1576 = llvm.urem %1575, %31  : i64
    %1577 = llvm.sub %1575, %1576 : i64
    %1578 = llvm.inttoptr %1577 : i64 to !llvm.ptr
    llvm.br ^bb341(%17 : i64)
  ^bb341(%1579: i64):  // 2 preds: ^bb340, ^bb348
    %1580 = llvm.icmp "slt" %1579, %30 : i64
    llvm.cond_br %1580, ^bb342, ^bb349
  ^bb342:  // pred: ^bb341
    llvm.br ^bb343(%17 : i64)
  ^bb343(%1581: i64):  // 2 preds: ^bb342, ^bb347
    %1582 = llvm.icmp "slt" %1581, %16 : i64
    llvm.cond_br %1582, ^bb344, ^bb348
  ^bb344:  // pred: ^bb343
    llvm.br ^bb345(%17 : i64)
  ^bb345(%1583: i64):  // 2 preds: ^bb344, ^bb346
    %1584 = llvm.icmp "slt" %1583, %35 : i64
    llvm.cond_br %1584, ^bb346, ^bb347
  ^bb346:  // pred: ^bb345
    %1585 = llvm.getelementptr %1578[%1579] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1586 = llvm.mul %1581, %30 : i64
    %1587 = llvm.add %1586, %1583 : i64
    %1588 = llvm.getelementptr %1585[%1587] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1588 : f32, !llvm.ptr
    %1589 = llvm.add %1583, %16 : i64
    llvm.br ^bb345(%1589 : i64)
  ^bb347:  // pred: ^bb345
    %1590 = llvm.add %1581, %16 : i64
    llvm.br ^bb343(%1590 : i64)
  ^bb348:  // pred: ^bb343
    %1591 = llvm.mul %165, %35 : i64
    %1592 = llvm.mul %1591, %75 : i64
    %1593 = llvm.getelementptr %1578[%1579] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1593, %1593, %1592) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1594 = llvm.add %1579, %35 : i64
    llvm.br ^bb341(%1594 : i64)
  ^bb349:  // pred: ^bb341
    llvm.br ^bb350(%17 : i64)
  ^bb350(%1595: i64):  // 2 preds: ^bb349, ^bb363
    %1596 = llvm.icmp "slt" %1595, %30 : i64
    llvm.cond_br %1596, ^bb351, ^bb364
  ^bb351:  // pred: ^bb350
    llvm.br ^bb352(%17 : i64)
  ^bb352(%1597: i64):  // 2 preds: ^bb351, ^bb362
    %1598 = llvm.icmp "slt" %1597, %34 : i64
    llvm.cond_br %1598, ^bb353, ^bb363
  ^bb353:  // pred: ^bb352
    llvm.br ^bb354(%17 : i64)
  ^bb354(%1599: i64):  // 2 preds: ^bb353, ^bb361
    %1600 = llvm.icmp "slt" %1599, %16 : i64
    llvm.cond_br %1600, ^bb355, ^bb362
  ^bb355:  // pred: ^bb354
    llvm.br ^bb356(%17 : i64)
  ^bb356(%1601: i64):  // 2 preds: ^bb355, ^bb360
    %1602 = llvm.icmp "slt" %1601, %35 : i64
    llvm.cond_br %1602, ^bb357, ^bb361
  ^bb357:  // pred: ^bb356
    llvm.br ^bb358(%17 : i64)
  ^bb358(%1603: i64):  // 2 preds: ^bb357, ^bb359
    %1604 = llvm.icmp "slt" %1603, %35 : i64
    llvm.cond_br %1604, ^bb359, ^bb360
  ^bb359:  // pred: ^bb358
    %1605 = llvm.getelementptr %1490[%1597] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1606 = llvm.mul %1599, %34 : i64
    %1607 = llvm.add %1606, %1603 : i64
    %1608 = llvm.getelementptr %1605[%1607] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1609 = llvm.load %1608 : !llvm.ptr -> f32
    %1610 = llvm.getelementptr %1578[%1595] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1611 = llvm.mul %1599, %30 : i64
    %1612 = llvm.add %1611, %1601 : i64
    %1613 = llvm.getelementptr %1610[%1612] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1614 = llvm.load %1613 : !llvm.ptr -> f32
    %1615 = llvm.fmul %1609, %39  : f32
    %1616 = llvm.fadd %1614, %1615  : f32
    llvm.store %1616, %1613 : f32, !llvm.ptr
    %1617 = llvm.add %1603, %16 : i64
    llvm.br ^bb358(%1617 : i64)
  ^bb360:  // pred: ^bb358
    %1618 = llvm.add %1601, %16 : i64
    llvm.br ^bb356(%1618 : i64)
  ^bb361:  // pred: ^bb356
    %1619 = llvm.add %1599, %16 : i64
    llvm.br ^bb354(%1619 : i64)
  ^bb362:  // pred: ^bb354
    %1620 = llvm.mul %165, %35 : i64
    %1621 = llvm.mul %1620, %75 : i64
    %1622 = llvm.getelementptr %1578[%1595] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1622, %1622, %1621) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1623 = llvm.add %1597, %35 : i64
    llvm.br ^bb352(%1623 : i64)
  ^bb363:  // pred: ^bb352
    %1624 = llvm.add %1595, %35 : i64
    llvm.br ^bb350(%1624 : i64)
  ^bb364:  // pred: ^bb350
    %1625 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %1626 = llvm.ptrtoint %1625 : !llvm.ptr to i64
    %1627 = llvm.add %1626, %57 : i64
    %1628 = llvm.urem %1627, %31  : i64
    %1629 = llvm.sub %1627, %1628 : i64
    %1630 = llvm.inttoptr %1629 : i64 to !llvm.ptr
    llvm.br ^bb365(%17 : i64)
  ^bb365(%1631: i64):  // 2 preds: ^bb364, ^bb372
    %1632 = llvm.icmp "slt" %1631, %34 : i64
    llvm.cond_br %1632, ^bb366, ^bb373
  ^bb366:  // pred: ^bb365
    llvm.br ^bb367(%17 : i64)
  ^bb367(%1633: i64):  // 2 preds: ^bb366, ^bb371
    %1634 = llvm.icmp "slt" %1633, %16 : i64
    llvm.cond_br %1634, ^bb368, ^bb372
  ^bb368:  // pred: ^bb367
    llvm.br ^bb369(%17 : i64)
  ^bb369(%1635: i64):  // 2 preds: ^bb368, ^bb370
    %1636 = llvm.icmp "slt" %1635, %35 : i64
    llvm.cond_br %1636, ^bb370, ^bb371
  ^bb370:  // pred: ^bb369
    %1637 = llvm.getelementptr %1630[%1631] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1638 = llvm.mul %1633, %34 : i64
    %1639 = llvm.add %1638, %1635 : i64
    %1640 = llvm.getelementptr %1637[%1639] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1640 : f32, !llvm.ptr
    %1641 = llvm.add %1635, %16 : i64
    llvm.br ^bb369(%1641 : i64)
  ^bb371:  // pred: ^bb369
    %1642 = llvm.add %1633, %16 : i64
    llvm.br ^bb367(%1642 : i64)
  ^bb372:  // pred: ^bb367
    %1643 = llvm.mul %165, %35 : i64
    %1644 = llvm.mul %1643, %75 : i64
    %1645 = llvm.getelementptr %1630[%1631] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1645, %1645, %1644) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1646 = llvm.add %1631, %35 : i64
    llvm.br ^bb365(%1646 : i64)
  ^bb373:  // pred: ^bb365
    llvm.br ^bb374(%17 : i64)
  ^bb374(%1647: i64):  // 2 preds: ^bb373, ^bb387
    %1648 = llvm.icmp "slt" %1647, %34 : i64
    llvm.cond_br %1648, ^bb375, ^bb388
  ^bb375:  // pred: ^bb374
    llvm.br ^bb376(%17 : i64)
  ^bb376(%1649: i64):  // 2 preds: ^bb375, ^bb386
    %1650 = llvm.icmp "slt" %1649, %30 : i64
    llvm.cond_br %1650, ^bb377, ^bb387
  ^bb377:  // pred: ^bb376
    llvm.br ^bb378(%17 : i64)
  ^bb378(%1651: i64):  // 2 preds: ^bb377, ^bb385
    %1652 = llvm.icmp "slt" %1651, %16 : i64
    llvm.cond_br %1652, ^bb379, ^bb386
  ^bb379:  // pred: ^bb378
    llvm.br ^bb380(%17 : i64)
  ^bb380(%1653: i64):  // 2 preds: ^bb379, ^bb384
    %1654 = llvm.icmp "slt" %1653, %35 : i64
    llvm.cond_br %1654, ^bb381, ^bb385
  ^bb381:  // pred: ^bb380
    llvm.br ^bb382(%17 : i64)
  ^bb382(%1655: i64):  // 2 preds: ^bb381, ^bb383
    %1656 = llvm.icmp "slt" %1655, %35 : i64
    llvm.cond_br %1656, ^bb383, ^bb384
  ^bb383:  // pred: ^bb382
    %1657 = llvm.getelementptr %1526[%1649] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1658 = llvm.mul %1651, %30 : i64
    %1659 = llvm.add %1658, %1655 : i64
    %1660 = llvm.getelementptr %1657[%1659] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1661 = llvm.load %1660 : !llvm.ptr -> f32
    %1662 = llvm.getelementptr %1578[%1649] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1663 = llvm.getelementptr %1662[%1659] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1664 = llvm.load %1663 : !llvm.ptr -> f32
    %1665 = llvm.getelementptr %1630[%1647] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1666 = llvm.mul %1651, %34 : i64
    %1667 = llvm.add %1666, %1653 : i64
    %1668 = llvm.getelementptr %1665[%1667] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1669 = llvm.load %1668 : !llvm.ptr -> f32
    %1670 = llvm.fneg %1661  : f32
    %1671 = llvm.intr.exp(%1670)  : (f32) -> f32
    %1672 = llvm.fadd %1661, %1671  : f32
    %1673 = llvm.fdiv %1661, %1672  : f32
    %1674 = llvm.fmul %1673, %1664  : f32
    %1675 = llvm.fmul %1674, %38  : f32
    %1676 = llvm.fadd %1669, %1675  : f32
    llvm.store %1676, %1668 : f32, !llvm.ptr
    %1677 = llvm.add %1655, %16 : i64
    llvm.br ^bb382(%1677 : i64)
  ^bb384:  // pred: ^bb382
    %1678 = llvm.add %1653, %16 : i64
    llvm.br ^bb380(%1678 : i64)
  ^bb385:  // pred: ^bb380
    %1679 = llvm.add %1651, %16 : i64
    llvm.br ^bb378(%1679 : i64)
  ^bb386:  // pred: ^bb378
    %1680 = llvm.mul %165, %35 : i64
    %1681 = llvm.mul %1680, %75 : i64
    %1682 = llvm.getelementptr %1630[%1647] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1682, %1682, %1681) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1683 = llvm.add %1649, %35 : i64
    llvm.br ^bb376(%1683 : i64)
  ^bb387:  // pred: ^bb376
    %1684 = llvm.add %1647, %35 : i64
    llvm.br ^bb374(%1684 : i64)
  ^bb388:  // pred: ^bb374
    %1685 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %1686 = llvm.ptrtoint %1685 : !llvm.ptr to i64
    %1687 = llvm.add %1686, %57 : i64
    %1688 = llvm.urem %1687, %31  : i64
    %1689 = llvm.sub %1687, %1688 : i64
    %1690 = llvm.inttoptr %1689 : i64 to !llvm.ptr
    %1691 = llvm.insertvalue %1685, %9[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1692 = llvm.insertvalue %1690, %1691[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1693 = llvm.insertvalue %17, %1692[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1694 = llvm.insertvalue %16, %1693[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1695 = llvm.insertvalue %34, %1694[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1696 = llvm.insertvalue %34, %1695[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1697 = llvm.insertvalue %16, %1696[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb389(%17 : i64)
  ^bb389(%1698: i64):  // 2 preds: ^bb388, ^bb396
    %1699 = llvm.icmp "slt" %1698, %34 : i64
    llvm.cond_br %1699, ^bb390, ^bb397
  ^bb390:  // pred: ^bb389
    llvm.br ^bb391(%17 : i64)
  ^bb391(%1700: i64):  // 2 preds: ^bb390, ^bb395
    %1701 = llvm.icmp "slt" %1700, %16 : i64
    llvm.cond_br %1701, ^bb392, ^bb396
  ^bb392:  // pred: ^bb391
    llvm.br ^bb393(%17 : i64)
  ^bb393(%1702: i64):  // 2 preds: ^bb392, ^bb394
    %1703 = llvm.icmp "slt" %1702, %35 : i64
    llvm.cond_br %1703, ^bb394, ^bb395
  ^bb394:  // pred: ^bb393
    %1704 = llvm.getelementptr %1432[%1698] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1705 = llvm.mul %1700, %34 : i64
    %1706 = llvm.add %1705, %1702 : i64
    %1707 = llvm.getelementptr %1704[%1706] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1708 = llvm.load %1707 : !llvm.ptr -> f32
    %1709 = llvm.getelementptr %1630[%1698] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1710 = llvm.getelementptr %1709[%1706] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1711 = llvm.load %1710 : !llvm.ptr -> f32
    %1712 = llvm.fadd %1708, %1711  : f32
    %1713 = llvm.getelementptr %1690[%1698] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1714 = llvm.getelementptr %1713[%1706] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1712, %1714 : f32, !llvm.ptr
    %1715 = llvm.add %1702, %16 : i64
    llvm.br ^bb393(%1715 : i64)
  ^bb395:  // pred: ^bb393
    %1716 = llvm.add %1700, %16 : i64
    llvm.br ^bb391(%1716 : i64)
  ^bb396:  // pred: ^bb391
    %1717 = llvm.mul %165, %35 : i64
    %1718 = llvm.mul %1717, %75 : i64
    %1719 = llvm.getelementptr %1690[%1698] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1719, %1719, %1718) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1720 = llvm.add %1698, %35 : i64
    llvm.br ^bb389(%1720 : i64)
  ^bb397:  // pred: ^bb389
    %1721 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %1722 = llvm.ptrtoint %1721 : !llvm.ptr to i64
    %1723 = llvm.add %1722, %57 : i64
    %1724 = llvm.urem %1723, %31  : i64
    %1725 = llvm.sub %1723, %1724 : i64
    %1726 = llvm.inttoptr %1725 : i64 to !llvm.ptr
    %1727 = llvm.insertvalue %1721, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1728 = llvm.insertvalue %1726, %1727[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1729 = llvm.insertvalue %17, %1728[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1730 = llvm.insertvalue %15, %1729[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1731 = llvm.insertvalue %32, %1730[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1732 = llvm.insertvalue %34, %1731[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1733 = llvm.insertvalue %6, %1732[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1734 = llvm.insertvalue %34, %1733[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1735 = llvm.insertvalue %16, %1734[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    "llvm.intr.memcpy"(%1726, %1053, %76) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1736 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %1737 = llvm.ptrtoint %1736 : !llvm.ptr to i64
    %1738 = llvm.add %1737, %57 : i64
    %1739 = llvm.urem %1738, %31  : i64
    %1740 = llvm.sub %1738, %1739 : i64
    %1741 = llvm.inttoptr %1740 : i64 to !llvm.ptr
    %1742 = llvm.insertvalue %1736, %7[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1743 = llvm.insertvalue %1741, %1742[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1744 = llvm.insertvalue %17, %1743[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1745 = llvm.insertvalue %15, %1744[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1746 = llvm.insertvalue %32, %1745[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1747 = llvm.insertvalue %34, %1746[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1748 = llvm.insertvalue %6, %1747[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1749 = llvm.insertvalue %34, %1748[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1750 = llvm.insertvalue %16, %1749[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    "llvm.intr.memcpy"(%1741, %1076, %76) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1751 = llvm.add %168, %16 : i64
    llvm.br ^bb3(%1751, %1697, %1735, %1750 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb398:  // pred: ^bb3
    %1752 = llvm.add %75, %31 : i64
    %1753 = llvm.call @malloc(%1752) : (i64) -> !llvm.ptr
    %1754 = llvm.ptrtoint %1753 : !llvm.ptr to i64
    %1755 = llvm.add %1754, %57 : i64
    %1756 = llvm.urem %1755, %31  : i64
    %1757 = llvm.sub %1755, %1756 : i64
    %1758 = llvm.inttoptr %1757 : i64 to !llvm.ptr
    llvm.br ^bb399(%17 : i64)
  ^bb399(%1759: i64):  // 2 preds: ^bb398, ^bb400
    %1760 = llvm.icmp "slt" %1759, %16 : i64
    llvm.cond_br %1760, ^bb400, ^bb401
  ^bb400:  // pred: ^bb399
    %1761 = llvm.getelementptr %1758[%1759] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1761 : f32, !llvm.ptr
    %1762 = llvm.add %1759, %16 : i64
    llvm.br ^bb399(%1762 : i64)
  ^bb401:  // pred: ^bb399
    llvm.br ^bb402(%17 : i64)
  ^bb402(%1763: i64):  // 2 preds: ^bb401, ^bb409
    %1764 = llvm.icmp "slt" %1763, %34 : i64
    llvm.cond_br %1764, ^bb403, ^bb410
  ^bb403:  // pred: ^bb402
    %1765 = llvm.extractvalue %169[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb404(%17 : i64)
  ^bb404(%1766: i64):  // 2 preds: ^bb403, ^bb408
    %1767 = llvm.icmp "slt" %1766, %16 : i64
    llvm.cond_br %1767, ^bb405, ^bb409
  ^bb405:  // pred: ^bb404
    llvm.br ^bb406(%17 : i64)
  ^bb406(%1768: i64):  // 2 preds: ^bb405, ^bb407
    %1769 = llvm.icmp "slt" %1768, %35 : i64
    llvm.cond_br %1769, ^bb407, ^bb408
  ^bb407:  // pred: ^bb406
    %1770 = llvm.getelementptr %1765[%1763] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1771 = llvm.mul %1766, %34 : i64
    %1772 = llvm.add %1771, %1768 : i64
    %1773 = llvm.getelementptr %1770[%1772] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1774 = llvm.load %1773 : !llvm.ptr -> f32
    %1775 = llvm.getelementptr %1758[%1766] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1776 = llvm.load %1775 : !llvm.ptr -> f32
    %1777 = llvm.fmul %1774, %1774  : f32
    %1778 = llvm.fadd %1776, %1777  : f32
    llvm.store %1778, %1775 : f32, !llvm.ptr
    %1779 = llvm.add %1768, %16 : i64
    llvm.br ^bb406(%1779 : i64)
  ^bb408:  // pred: ^bb406
    %1780 = llvm.add %1766, %16 : i64
    llvm.br ^bb404(%1780 : i64)
  ^bb409:  // pred: ^bb404
    %1781 = llvm.add %1763, %35 : i64
    llvm.br ^bb402(%1781 : i64)
  ^bb410:  // pred: ^bb402
    %1782 = llvm.getelementptr %48[32000] : (!llvm.ptr) -> !llvm.ptr, f32
    %1783 = llvm.ptrtoint %1782 : !llvm.ptr to i64
    %1784 = llvm.add %1783, %31 : i64
    %1785 = llvm.call @malloc(%1784) : (i64) -> !llvm.ptr
    %1786 = llvm.ptrtoint %1785 : !llvm.ptr to i64
    %1787 = llvm.add %1786, %57 : i64
    %1788 = llvm.urem %1787, %31  : i64
    %1789 = llvm.sub %1787, %1788 : i64
    %1790 = llvm.inttoptr %1789 : i64 to !llvm.ptr
    %1791 = llvm.insertvalue %1785, %9[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1792 = llvm.insertvalue %1790, %1791[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1793 = llvm.insertvalue %17, %1792[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1794 = llvm.insertvalue %16, %1793[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1795 = llvm.insertvalue %29, %1794[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1796 = llvm.insertvalue %29, %1795[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %1797 = llvm.insertvalue %16, %1796[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb411(%17 : i64)
  ^bb411(%1798: i64):  // 2 preds: ^bb410, ^bb418
    %1799 = llvm.icmp "slt" %1798, %29 : i64
    llvm.cond_br %1799, ^bb412, ^bb419
  ^bb412:  // pred: ^bb411
    llvm.br ^bb413(%17 : i64)
  ^bb413(%1800: i64):  // 2 preds: ^bb412, ^bb417
    %1801 = llvm.icmp "slt" %1800, %16 : i64
    llvm.cond_br %1801, ^bb414, ^bb418
  ^bb414:  // pred: ^bb413
    llvm.br ^bb415(%17 : i64)
  ^bb415(%1802: i64):  // 2 preds: ^bb414, ^bb416
    %1803 = llvm.icmp "slt" %1802, %35 : i64
    llvm.cond_br %1803, ^bb416, ^bb417
  ^bb416:  // pred: ^bb415
    %1804 = llvm.getelementptr %1790[%1798] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1805 = llvm.mul %1800, %29 : i64
    %1806 = llvm.add %1805, %1802 : i64
    %1807 = llvm.getelementptr %1804[%1806] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %1807 : f32, !llvm.ptr
    %1808 = llvm.add %1802, %16 : i64
    llvm.br ^bb415(%1808 : i64)
  ^bb417:  // pred: ^bb415
    %1809 = llvm.add %1800, %16 : i64
    llvm.br ^bb413(%1809 : i64)
  ^bb418:  // pred: ^bb413
    %1810 = llvm.mul %165, %35 : i64
    %1811 = llvm.mul %1810, %75 : i64
    %1812 = llvm.getelementptr %1790[%1798] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1812, %1812, %1811) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1813 = llvm.add %1798, %35 : i64
    llvm.br ^bb411(%1813 : i64)
  ^bb419:  // pred: ^bb411
    llvm.br ^bb420(%17 : i64)
  ^bb420(%1814: i64):  // 2 preds: ^bb419, ^bb433
    %1815 = llvm.icmp "slt" %1814, %29 : i64
    llvm.cond_br %1815, ^bb421, ^bb434
  ^bb421:  // pred: ^bb420
    llvm.br ^bb422(%17 : i64)
  ^bb422(%1816: i64):  // 2 preds: ^bb421, ^bb432
    %1817 = llvm.icmp "slt" %1816, %34 : i64
    llvm.cond_br %1817, ^bb423, ^bb433
  ^bb423:  // pred: ^bb422
    %1818 = llvm.extractvalue %169[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb424(%17 : i64)
  ^bb424(%1819: i64):  // 2 preds: ^bb423, ^bb431
    %1820 = llvm.icmp "slt" %1819, %16 : i64
    llvm.cond_br %1820, ^bb425, ^bb432
  ^bb425:  // pred: ^bb424
    llvm.br ^bb426(%17 : i64)
  ^bb426(%1821: i64):  // 2 preds: ^bb425, ^bb430
    %1822 = llvm.icmp "slt" %1821, %35 : i64
    llvm.cond_br %1822, ^bb427, ^bb431
  ^bb427:  // pred: ^bb426
    llvm.br ^bb428(%17 : i64)
  ^bb428(%1823: i64):  // 2 preds: ^bb427, ^bb429
    %1824 = llvm.icmp "slt" %1823, %35 : i64
    llvm.cond_br %1824, ^bb429, ^bb430
  ^bb429:  // pred: ^bb428
    %1825 = llvm.getelementptr %1818[%1816] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1826 = llvm.mul %1819, %34 : i64
    %1827 = llvm.add %1826, %1823 : i64
    %1828 = llvm.getelementptr %1825[%1827] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1829 = llvm.load %1828 : !llvm.ptr -> f32
    %1830 = llvm.getelementptr %1758[%1819] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1831 = llvm.load %1830 : !llvm.ptr -> f32
    %1832 = llvm.getelementptr %1790[%1814] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1833 = llvm.mul %1819, %29 : i64
    %1834 = llvm.add %1833, %1821 : i64
    %1835 = llvm.getelementptr %1832[%1834] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1836 = llvm.load %1835 : !llvm.ptr -> f32
    %1837 = llvm.fdiv %1831, %28  : f32
    %1838 = llvm.fadd %1837, %22  : f32
    %1839 = llvm.intr.sqrt(%1838)  : (f32) -> f32
    %1840 = llvm.fdiv %18, %1839  : f32
    %1841 = llvm.fmul %1829, %1840  : f32
    %1842 = llvm.fmul %1841, %37  : f32
    %1843 = llvm.fmul %1842, %36  : f32
    %1844 = llvm.fadd %1836, %1843  : f32
    llvm.store %1844, %1835 : f32, !llvm.ptr
    %1845 = llvm.add %1823, %16 : i64
    llvm.br ^bb428(%1845 : i64)
  ^bb430:  // pred: ^bb428
    %1846 = llvm.add %1821, %16 : i64
    llvm.br ^bb426(%1846 : i64)
  ^bb431:  // pred: ^bb426
    %1847 = llvm.add %1819, %16 : i64
    llvm.br ^bb424(%1847 : i64)
  ^bb432:  // pred: ^bb424
    %1848 = llvm.mul %165, %35 : i64
    %1849 = llvm.mul %1848, %75 : i64
    %1850 = llvm.getelementptr %1790[%1814] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%1850, %1850, %1849) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1851 = llvm.add %1816, %35 : i64
    llvm.br ^bb422(%1851 : i64)
  ^bb433:  // pred: ^bb422
    %1852 = llvm.add %1814, %35 : i64
    llvm.br ^bb420(%1852 : i64)
  ^bb434:  // pred: ^bb420
    %1853 = llvm.call @malloc(%1752) : (i64) -> !llvm.ptr
    %1854 = llvm.ptrtoint %1853 : !llvm.ptr to i64
    %1855 = llvm.add %1854, %57 : i64
    %1856 = llvm.urem %1855, %31  : i64
    %1857 = llvm.sub %1855, %1856 : i64
    %1858 = llvm.inttoptr %1857 : i64 to !llvm.ptr
    llvm.br ^bb435(%17 : i64)
  ^bb435(%1859: i64):  // 2 preds: ^bb434, ^bb436
    %1860 = llvm.icmp "slt" %1859, %16 : i64
    llvm.cond_br %1860, ^bb436, ^bb437
  ^bb436:  // pred: ^bb435
    %1861 = llvm.getelementptr %1858[%1859] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %43, %1861 : f32, !llvm.ptr
    %1862 = llvm.add %1859, %16 : i64
    llvm.br ^bb435(%1862 : i64)
  ^bb437:  // pred: ^bb435
    %1863 = llvm.getelementptr %48[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %1864 = llvm.ptrtoint %1863 : !llvm.ptr to i64
    %1865 = llvm.add %1864, %31 : i64
    %1866 = llvm.call @malloc(%1865) : (i64) -> !llvm.ptr
    %1867 = llvm.ptrtoint %1866 : !llvm.ptr to i64
    %1868 = llvm.add %1867, %57 : i64
    %1869 = llvm.urem %1868, %31  : i64
    %1870 = llvm.sub %1868, %1869 : i64
    %1871 = llvm.inttoptr %1870 : i64 to !llvm.ptr
    llvm.br ^bb438(%17 : i64)
  ^bb438(%1872: i64):  // 2 preds: ^bb437, ^bb439
    %1873 = llvm.icmp "slt" %1872, %16 : i64
    llvm.cond_br %1873, ^bb439, ^bb440
  ^bb439:  // pred: ^bb438
    %1874 = llvm.getelementptr %1871[%1872] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %13, %1874 : i64, !llvm.ptr
    %1875 = llvm.add %1872, %16 : i64
    llvm.br ^bb438(%1875 : i64)
  ^bb440:  // pred: ^bb438
    llvm.br ^bb441(%17 : i64)
  ^bb441(%1876: i64):  // 2 preds: ^bb440, ^bb448
    %1877 = llvm.icmp "slt" %1876, %29 : i64
    llvm.cond_br %1877, ^bb442, ^bb449
  ^bb442:  // pred: ^bb441
    llvm.br ^bb443(%17 : i64)
  ^bb443(%1878: i64):  // 2 preds: ^bb442, ^bb447
    %1879 = llvm.icmp "slt" %1878, %16 : i64
    llvm.cond_br %1879, ^bb444, ^bb448
  ^bb444:  // pred: ^bb443
    llvm.br ^bb445(%17 : i64)
  ^bb445(%1880: i64):  // 2 preds: ^bb444, ^bb446
    %1881 = llvm.icmp "slt" %1880, %35 : i64
    llvm.cond_br %1881, ^bb446, ^bb447
  ^bb446:  // pred: ^bb445
    %1882 = llvm.getelementptr %1790[%1876] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1883 = llvm.mul %1878, %29 : i64
    %1884 = llvm.add %1883, %1880 : i64
    %1885 = llvm.getelementptr %1882[%1884] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1886 = llvm.load %1885 : !llvm.ptr -> f32
    %1887 = llvm.getelementptr %1858[%1878] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1888 = llvm.load %1887 : !llvm.ptr -> f32
    %1889 = llvm.getelementptr %1871[%1878] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %1890 = llvm.load %1889 : !llvm.ptr -> i64
    %1891 = llvm.add %1880, %1876 : i64
    %1892 = llvm.fcmp "ogt" %1886, %1888 : f32
    %1893 = llvm.select %1892, %1886, %1888 : i1, f32
    %1894 = llvm.select %1892, %1891, %1890 : i1, i64
    llvm.store %1893, %1887 : f32, !llvm.ptr
    llvm.store %1894, %1889 : i64, !llvm.ptr
    %1895 = llvm.add %1880, %16 : i64
    llvm.br ^bb445(%1895 : i64)
  ^bb447:  // pred: ^bb445
    %1896 = llvm.add %1878, %16 : i64
    llvm.br ^bb443(%1896 : i64)
  ^bb448:  // pred: ^bb443
    %1897 = llvm.add %1876, %35 : i64
    llvm.br ^bb441(%1897 : i64)
  ^bb449:  // pred: ^bb441
    %1898 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %1797, %1898 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.call @printMemrefF32(%11, %1898) : (i64, !llvm.ptr) -> ()
    %1899 = llvm.add %146, %14 : i64
    llvm.br ^bb1(%1899, %170, %171 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb450:  // pred: ^bb1
    llvm.return
  }
}
