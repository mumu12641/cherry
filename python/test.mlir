"builtin.module"() ({
  "cherry.func"() <{function_type = (i64, i64, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[32000x768xf32]>, !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>), sym_name = "llama_forward"}> ({
  ^bb0(%arg8: i64, %arg9: i64, %arg10: !cherry.cherry_tensor<[?xf32]>, %arg11: !cherry.cherry_tensor<[?xf32]>, %arg12: !cherry.cherry_tensor<[32000x768xf32]>, %arg13: !cherry.cherry_tensor<[12x768xf32]>, %arg14: !cherry.cherry_tensor<[12x768x768xf32]>, %arg15: !cherry.cherry_tensor<[12x768x768xf32]>, %arg16: !cherry.cherry_tensor<[12x768x768xf32]>, %arg17: !cherry.cherry_tensor<[12x768x768xf32]>, %arg18: !cherry.cherry_tensor<[12x768xf32]>, %arg19: !cherry.cherry_tensor<[12x768x2048xf32]>, %arg20: !cherry.cherry_tensor<[12x2048x768xf32]>, %arg21: !cherry.cherry_tensor<[12x768x2048xf32]>, %arg22: !cherry.cherry_tensor<[768xf32]>, %arg23: !cherry.cherry_tensor<[768x32000xf32]>):
    %26 = "cherry.constant"() <{value = 0 : i64}> : () -> i64
    %27 = "cherry.constant"() <{value = 1 : i64}> : () -> i64
    %28 = "cherry.constant"() <{value = 768 : i64}> : () -> i64
    %29 = "cherry.constant"() <{value = 2048 : i64}> : () -> i64
    %30 = "cherry.constant"() <{value = 12 : i64}> : () -> i64
    %31 = "cherry.constant"() <{value = 64 : i64}> : () -> i64
    %32 = "cherry.constant"() <{value = 1.250000e-01 : f32}> : () -> f32
    %33 = "cherry.constant"() <{value = 1024 : i64}> : () -> i64
    %34 = "cherry.constant"() <{value = 32000 : i64}> : () -> i64
    %35 = "cherry.scalar_add"(%arg9, %27) : (i64, i64) -> i64
    %36 = "cherry.tensor_slice"(%arg12, %arg8, %26) <{sizes = [1, 768], squeeze = false}> : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    %37 = "arith.constant"() <{value = 0 : index}> : () -> index
    %38 = "arith.constant"() <{value = 12 : index}> : () -> index
    %39 = "arith.constant"() <{value = 1 : index}> : () -> index
    %40:3 = "scf.for"(%37, %38, %39, %36, %arg10, %arg11) ({
    ^bb0(%arg24: index, %arg25: !cherry.cherry_tensor<[?xf32]>, %arg26: !cherry.cherry_tensor<[?xf32]>, %arg27: !cherry.cherry_tensor<[?xf32]>):
      %43 = "arith.index_cast"(%arg24) : (index) -> i64
      %44 = "cherry.tensor_slice"(%arg13, %43, %26) <{sizes = [1, 768], squeeze = true}> : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %45 = "cherry.rmsnorm"(%arg25, %44) <{epsilon = 9.99999974E-6 : f32}> : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %46 = "cherry.tensor_slice"(%arg14, %43, %26, %26) <{sizes = [1, 768, 768], squeeze = true}> : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %47 = "cherry.tensor_slice"(%arg15, %43, %26, %26) <{sizes = [1, 768, 768], squeeze = true}> : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %48 = "cherry.tensor_slice"(%arg16, %43, %26, %26) <{sizes = [1, 768, 768], squeeze = true}> : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %49 = "cherry.matmul"(%45, %46) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %50 = "cherry.matmul"(%45, %47) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %51 = "cherry.matmul"(%45, %48) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %52 = "cherry.reshape"(%49) <{new_shape = [1, 12, 64]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %53 = "cherry.rope"(%52, %arg9) : (!cherry.cherry_tensor<[?xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
      %54 = "cherry.reshape"(%53) <{new_shape = [1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %55 = "cherry.reshape"(%50) <{new_shape = [1, 12, 64]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %56 = "cherry.rope"(%55, %arg9) : (!cherry.cherry_tensor<[?xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
      %57 = "cherry.reshape"(%56) <{new_shape = [1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %58 = "cherry.reshape"(%57) <{new_shape = [1, 1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %59 = "cherry.tensor_set_slice"(%arg26, %58, %43, %arg9) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %60 = "cherry.reshape"(%51) <{new_shape = [1, 1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %61 = "cherry.tensor_set_slice"(%arg27, %60, %43, %arg9) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %62 = "cherry.tensor_slice"(%59, %43, %26, %26) <{sizes = [1, 1024, 768], squeeze = true}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %63 = "cherry.tensor_slice"(%61, %43, %26, %26) <{sizes = [1, 1024, 768], squeeze = true}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %64 = "cherry.create_tensor"() <{value = dense<0.000000e+00> : tensor<1x12x64xf32>}> : () -> !cherry.cherry_tensor<[?xf32]>
      %65 = "arith.constant"() <{value = 0 : index}> : () -> index
      %66 = "arith.constant"() <{value = 12 : index}> : () -> index
      %67 = "arith.constant"() <{value = 1 : index}> : () -> index
      %68 = "scf.for"(%65, %66, %67, %64) ({
      ^bb0(%arg28: index, %arg29: !cherry.cherry_tensor<[?xf32]>):
        %84 = "arith.index_cast"(%arg28) : (index) -> i64
        %85 = "cherry.scalar_mul"(%84, %31) : (i64, i64) -> i64
        %86 = "cherry.tensor_slice"(%54, %26, %85) <{sizes = [1, 64], squeeze = false}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
        %87 = "cherry.tensor_slice"(%62, %26, %85) <{sizes = [1024, 64], squeeze = false}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
        %88 = "cherry.transpose"(%87) <{permutation = [1, 0]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
        %89 = "cherry.masked_matmul"(%86, %88, %35) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
        %90 = "cherry.tensor_mul_scalar"(%89, %32) : (!cherry.cherry_tensor<[?xf32]>, f32) -> !cherry.cherry_tensor<[?xf32]>
        %91 = "cherry.softmax"(%90) <{axis = 1 : i64}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
        %92 = "cherry.tensor_slice"(%63, %26, %85) <{sizes = [1024, 64], squeeze = false}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
        %93 = "cherry.matmul"(%91, %92) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
        %94 = "cherry.reshape"(%93) <{new_shape = [1, 1, 64]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
        %95 = "cherry.tensor_set_slice"(%arg29, %94, %26, %84) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
        "scf.yield"(%95) : (!cherry.cherry_tensor<[?xf32]>) -> ()
      }) : (index, index, index, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %69 = "cherry.reshape"(%68) <{new_shape = [1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %70 = "cherry.tensor_slice"(%arg17, %43, %26, %26) <{sizes = [1, 768, 768], squeeze = true}> : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %71 = "cherry.matmul"(%69, %70) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %72 = "cherry.tensor_add"(%arg25, %71) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %73 = "cherry.tensor_slice"(%arg18, %43, %26) <{sizes = [1, 768], squeeze = true}> : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %74 = "cherry.rmsnorm"(%72, %73) <{epsilon = 9.99999974E-6 : f32}> : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %75 = "cherry.tensor_slice"(%arg19, %43, %26, %26) <{sizes = [1, 768, 2048], squeeze = true}> : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %76 = "cherry.tensor_slice"(%arg21, %43, %26, %26) <{sizes = [1, 768, 2048], squeeze = true}> : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %77 = "cherry.matmul"(%74, %75) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %78 = "cherry.matmul"(%74, %76) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %79 = "cherry.tensor_silu"(%77) : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %80 = "cherry.tensor_mul"(%79, %78) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %81 = "cherry.tensor_slice"(%arg20, %43, %26, %26) <{sizes = [1, 2048, 768], squeeze = true}> : (!cherry.cherry_tensor<[12x2048x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %82 = "cherry.matmul"(%80, %81) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %83 = "cherry.tensor_add"(%72, %82) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      "scf.yield"(%83, %59, %61) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> ()
    }) : (index, index, index, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>)
    %41 = "cherry.rmsnorm"(%40#0, %arg22) <{epsilon = 9.99999974E-6 : f32}> : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[768xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %42 = "cherry.matmul"(%41, %arg23) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    "cherry.return"(%42, %40#1, %40#2) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "cherry.func"() <{function_type = () -> (), sym_name = "host"}> ({
    %0 = "cherry.constant"() <{value = 32000 : i64}> : () -> i64
    "cherry.runtime_call"(%0) <{callee = "build_tokenizer", str_args = ["/home/nx/ycy/pb/cherry/tests/llama/tokenizer.bin"]}> : (i64) -> ()
    %1 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/token_embeddings.bin", shape = [32000, 768]}> : () -> !cherry.cherry_tensor<[32000x768xf32]>
    %2 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_rms_att_weight.bin", shape = [12, 768]}> : () -> !cherry.cherry_tensor<[12x768xf32]>
    %3 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_wq.bin", shape = [12, 768, 768]}> : () -> !cherry.cherry_tensor<[12x768x768xf32]>
    %4 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_wk.bin", shape = [12, 768, 768]}> : () -> !cherry.cherry_tensor<[12x768x768xf32]>
    %5 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_wv.bin", shape = [12, 768, 768]}> : () -> !cherry.cherry_tensor<[12x768x768xf32]>
    %6 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_wo.bin", shape = [12, 768, 768]}> : () -> !cherry.cherry_tensor<[12x768x768xf32]>
    %7 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_rms_ffn_weight.bin", shape = [12, 768]}> : () -> !cherry.cherry_tensor<[12x768xf32]>
    %8 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_w1.bin", shape = [12, 768, 2048]}> : () -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %9 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_w2.bin", shape = [12, 2048, 768]}> : () -> !cherry.cherry_tensor<[12x2048x768xf32]>
    %10 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/layers_w3.bin", shape = [12, 768, 2048]}> : () -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %11 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/final_rms_norm.bin", shape = [768]}> : () -> !cherry.cherry_tensor<[768xf32]>
    %12 = "cherry.weight"() <{elem_type = f32, path = "/home/nx/ycy/pb/cherry/utils/stories110M/output_wcls.bin", shape = [768, 32000]}> : () -> !cherry.cherry_tensor<[768x32000xf32]>
    %13 = "cherry.create_tensor"() <{value = dense<0.000000e+00> : tensor<12x1024x768xf32>}> : () -> !cherry.cherry_tensor<[?xf32]>
    %14 = "cherry.create_tensor"() <{value = dense<0.000000e+00> : tensor<12x1024x768xf32>}> : () -> !cherry.cherry_tensor<[?xf32]>
    %15 = "cherry.constant"() <{value = 1 : i64}> : () -> i64
    %16 = "cherry.constant"() <{value = 0 : i64}> : () -> i64
    %17 = "cherry.constant"() <{value = 128 : i64}> : () -> i64
    "cherry.runtime_call"() <{callee = "start"}> : () -> ()
    %18:4 = "scf.while"(%15, %16, %13, %14) ({
    ^bb0(%arg4: i64, %arg5: i64, %arg6: !cherry.cherry_tensor<[?xf32]>, %arg7: !cherry.cherry_tensor<[?xf32]>):
      %25 = "arith.cmpi"(%arg5, %17) <{predicate = 2 : i64}> : (i64, i64) -> i1
      "scf.condition"(%25, %arg4, %arg5, %arg6, %arg7) : (i1, i64, i64, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> ()
    }, {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[?xf32]>, %arg3: !cherry.cherry_tensor<[?xf32]>):
      %19 = "cherry.constant"() <{value = 1 : i64}> : () -> i64
      %20 = "cherry.scalar_add"(%arg1, %19) : (i64, i64) -> i64
      %21:3 = "cherry.call"(%arg0, %arg1, %arg2, %arg3, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12) <{callee = @llama_forward}> : (i64, i64, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[32000x768xf32]>, !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>)
      %22 = "cherry.argmax"(%21#0) <{dim = 1 : i64}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xi64]>
      %23 = "cherry.constant"() <{value = 0 : i64}> : () -> i64
      %24 = "cherry.tensor_get"(%22, %23) : (!cherry.cherry_tensor<[?xi64]>, i64) -> i64
      "cherry.runtime_call"(%arg0, %24) <{callee = "decode"}> : (i64, i64) -> ()
      "scf.yield"(%24, %20, %21#1, %21#2) : (i64, i64, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> ()
    }) : (i64, i64, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> (i64, i64, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>)
    "cherry.runtime_call"(%17) <{callee = "end"}> : (i64) -> ()
    "cherry.runtime_call"() <{callee = "free_tokenizer"}> : () -> ()
    "cherry.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
