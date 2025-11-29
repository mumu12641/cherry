"builtin.module"() ({
  "func.func"() <{function_type = (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>, sym_name = "simple_transformer_block"}> ({
  ^bb0(%arg0: !cherry.cherry_tensor<[?x?x8xf32]>, %arg1: !cherry.cherry_tensor<[8x8xf32]>, %arg2: !cherry.cherry_tensor<[8x8xf32]>, %arg3: !cherry.cherry_tensor<[8x8xf32]>, %arg4: !cherry.cherry_tensor<[8x32xf32]>, %arg5: !cherry.cherry_tensor<[32x8xf32]>, %arg6: !cherry.cherry_tensor<[8xf32]>, %arg7: !cherry.cherry_tensor<[8xf32]>):
    %9 = "cherry.matmul"(%arg0, %arg1) : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %10 = "cherry.matmul"(%arg0, %arg2) : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %11 = "cherry.matmul"(%arg0, %arg3) : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %12 = "cherry.constant"() <{value = 0 : i64}> : () -> i64
    %13 = "cherry.constant"() <{value = 2 : i64}> : () -> i64
    %14 = "cherry.constant"() <{value = 1 : i64}> : () -> i64
    %15 = "cherry.transpose"(%10, %12, %13, %14) : (!cherry.cherry_tensor<[?x?x8xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?x8x?xf32]>
    %16 = "cherry.matmul"(%9, %15) : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x8x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %17 = "cherry.create_tensor"() <{value = dense<2.828400e+00> : tensor<1xf32>}> : () -> !cherry.cherry_tensor<[1xf32]>
    %18 = "cherry.constant"() <{value = 1 : i64}> : () -> i64
    %19 = "cherry.constant"() <{value = 4 : i64}> : () -> i64
    %20 = "cherry.broadcast"(%17, %18, %19, %19) : (!cherry.cherry_tensor<[1xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %21 = "cherry.tensor_div"(%16, %20) : (!cherry.cherry_tensor<[?x?x?xf32]>, !cherry.cherry_tensor<[?x?x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %22 = "cherry.softmax"(%21) <{axis = 2 : i64}> : (!cherry.cherry_tensor<[?x?x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %23 = "cherry.matmul"(%22, %11) : (!cherry.cherry_tensor<[?x?x?xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %24 = "cherry.tensor_add"(%arg0, %23) : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %25 = "cherry.layernorm"(%24, %arg6, %arg7) <{eps = 9.99999974E-6 : f32}> : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %26 = "cherry.matmul"(%25, %arg4) : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[?x?x32xf32]>
    %27 = "cherry.tensor_relu"(%26) : (!cherry.cherry_tensor<[?x?x32xf32]>) -> !cherry.cherry_tensor<[?x?x32xf32]>
    %28 = "cherry.matmul"(%27, %arg5) : (!cherry.cherry_tensor<[?x?x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %29 = "cherry.tensor_add"(%25, %28) : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %30 = "cherry.layernorm"(%29, %arg6, %arg7) <{eps = 9.99999974E-6 : f32}> : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    "func.return"(%30) : (!cherry.cherry_tensor<[?x?x8xf32]>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "cherry.create_tensor"() <{value = dense<5.000000e-01> : tensor<1x4x8xf32>}> : () -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = "cherry.create_tensor"() <{value = dense<1.000000e-01> : tensor<8x8xf32>}> : () -> !cherry.cherry_tensor<[8x8xf32]>
    %2 = "cherry.create_tensor"() <{value = dense<1.000000e-01> : tensor<8x8xf32>}> : () -> !cherry.cherry_tensor<[8x8xf32]>
    %3 = "cherry.create_tensor"() <{value = dense<1.000000e-01> : tensor<8x8xf32>}> : () -> !cherry.cherry_tensor<[8x8xf32]>
    %4 = "cherry.create_tensor"() <{value = dense<2.000000e-01> : tensor<8x32xf32>}> : () -> !cherry.cherry_tensor<[8x32xf32]>
    %5 = "cherry.create_tensor"() <{value = dense<2.000000e-01> : tensor<32x8xf32>}> : () -> !cherry.cherry_tensor<[32x8xf32]>
    %6 = "cherry.create_tensor"() <{value = dense<1.000000e+00> : tensor<8xf32>}> : () -> !cherry.cherry_tensor<[8xf32]>
    %7 = "cherry.create_tensor"() <{value = dense<0.000000e+00> : tensor<8xf32>}> : () -> !cherry.cherry_tensor<[8xf32]>
    %8 = "func.call"(%0, %1, %2, %3, %4, %5, %6, %7) <{callee = @simple_transformer_block}> : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
