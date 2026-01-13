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
    %36 = "cherry.tensor_slice"(%arg12, %arg8, %26) <{sizes = [1, 768]}> : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    %37 = "arith.constant"() <{value = 0 : index}> : () -> index
    %38 = "arith.constant"() <{value = 12 : index}> : () -> index
    %39 = "arith.constant"() <{value = 1 : index}> : () -> index
    %40:3 = "scf.for"(%37, %38, %39, %36, %arg10, %arg11) ({
    ^bb0(%arg24: index, %arg25: !cherry.cherry_tensor<[?xf32]>, %arg26: !cherry.cherry_tensor<[?xf32]>, %arg27: !cherry.cherry_tensor<[?xf32]>):
      %43 = "arith.index_cast"(%arg24) : (index) -> i64
      %44 = "cherry.tensor_slice"(%arg13, %43, %26) <{sizes = [1, 768]}> : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %45 = "cherry.reshape"(%44) <{new_shape = [768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %46 = "cherry.rmsnorm"(%arg25, %45) <{epsilon = 9.99999974E-6 : f32}> : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %47 = "cherry.tensor_slice"(%arg14, %43, %26, %26) <{sizes = [1, 768, 768]}> : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %48 = "cherry.reshape"(%47) <{new_shape = [768, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %49 = "cherry.tensor_slice"(%arg15, %43, %26, %26) <{sizes = [1, 768, 768]}> : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %50 = "cherry.reshape"(%49) <{new_shape = [768, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %51 = "cherry.tensor_slice"(%arg16, %43, %26, %26) <{sizes = [1, 768, 768]}> : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %52 = "cherry.reshape"(%51) <{new_shape = [768, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %53 = "cherry.matmul"(%46, %48) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %54 = "cherry.matmul"(%46, %50) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %55 = "cherry.matmul"(%46, %52) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %56 = "cherry.reshape"(%53) <{new_shape = [1, 12, 64]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %57 = "cherry.rope"(%56, %arg9) : (!cherry.cherry_tensor<[?xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
      %58 = "cherry.reshape"(%57) <{new_shape = [1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %59 = "cherry.reshape"(%54) <{new_shape = [1, 12, 64]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %60 = "cherry.rope"(%59, %arg9) : (!cherry.cherry_tensor<[?xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
      %61 = "cherry.reshape"(%60) <{new_shape = [1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %62 = "cherry.reshape"(%61) <{new_shape = [1, 1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %63 = "cherry.tensor_set_slice"(%arg26, %62, %43, %arg9) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %64 = "cherry.reshape"(%55) <{new_shape = [1, 1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %65 = "cherry.tensor_set_slice"(%arg27, %64, %43, %arg9) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %66 = "cherry.tensor_slice"(%63, %43, %26, %26) <{sizes = [1, 1024, 768]}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %67 = "cherry.reshape"(%66) <{new_shape = [1024, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %68 = "cherry.tensor_slice"(%65, %43, %26, %26) <{sizes = [1, 1024, 768]}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %69 = "cherry.reshape"(%68) <{new_shape = [1024, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %70 = "cherry.create_tensor"() <{value = dense<0.000000e+00> : tensor<1x12x64xf32>}> : () -> !cherry.cherry_tensor<[?xf32]>
      %71 = "arith.constant"() <{value = 0 : index}> : () -> index
      %72 = "arith.constant"() <{value = 12 : index}> : () -> index
      %73 = "arith.constant"() <{value = 1 : index}> : () -> index
      %74 = "scf.for"(%71, %72, %73, %70) ({
      ^bb0(%arg28: index, %arg29: !cherry.cherry_tensor<[?xf32]>):
        %95 = "arith.index_cast"(%arg28) : (index) -> i64
        %96 = "cherry.scalar_mul"(%95, %31) : (i64, i64) -> i64
        %97 = "cherry.tensor_slice"(%58, %26, %96) <{sizes = [1, 64]}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
        %98 = "cherry.tensor_slice"(%67, %26, %96) <{sizes = [1024, 64]}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
        %99 = "cherry.transpose"(%98) <{permutation = [1, 0]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
        %100 = "cherry.masked_matmul"(%97, %99, %35) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
        %101 = "cherry.tensor_mul_scalar"(%100, %32) : (!cherry.cherry_tensor<[?xf32]>, f32) -> !cherry.cherry_tensor<[?xf32]>
        %102 = "cherry.softmax"(%101) <{axis = 1 : i64}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
        %103 = "cherry.tensor_slice"(%69, %26, %96) <{sizes = [1024, 64]}> : (!cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
        %104 = "cherry.matmul"(%102, %103) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
        %105 = "cherry.reshape"(%104) <{new_shape = [1, 1, 64]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
        %106 = "cherry.tensor_set_slice"(%arg29, %105, %26, %95) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
        "scf.yield"(%106) : (!cherry.cherry_tensor<[?xf32]>) -> ()
      }) : (index, index, index, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %75 = "cherry.reshape"(%74) <{new_shape = [1, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %76 = "cherry.tensor_slice"(%arg17, %43, %26, %26) <{sizes = [1, 768, 768]}> : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %77 = "cherry.reshape"(%76) <{new_shape = [768, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %78 = "cherry.matmul"(%75, %77) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %79 = "cherry.tensor_add"(%arg25, %78) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %80 = "cherry.tensor_slice"(%arg18, %43, %26) <{sizes = [1, 768]}> : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %81 = "cherry.reshape"(%80) <{new_shape = [768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %82 = "cherry.rmsnorm"(%79, %81) <{epsilon = 9.99999974E-6 : f32}> : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %83 = "cherry.tensor_slice"(%arg19, %43, %26, %26) <{sizes = [1, 768, 2048]}> : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %84 = "cherry.reshape"(%83) <{new_shape = [768, 2048]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %85 = "cherry.tensor_slice"(%arg21, %43, %26, %26) <{sizes = [1, 768, 2048]}> : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %86 = "cherry.reshape"(%85) <{new_shape = [768, 2048]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %87 = "cherry.matmul"(%82, %84) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %88 = "cherry.matmul"(%82, %86) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %89 = "cherry.tensor_silu"(%87) : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %90 = "cherry.tensor_mul"(%89, %88) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %91 = "cherry.tensor_slice"(%arg20, %43, %26, %26) <{sizes = [1, 2048, 768]}> : (!cherry.cherry_tensor<[12x2048x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %92 = "cherry.reshape"(%91) <{new_shape = [2048, 768]}> : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %93 = "cherry.matmul"(%90, %92) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %94 = "cherry.tensor_add"(%79, %93) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      "scf.yield"(%94, %63, %65) : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> ()
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
    %17 = "cherry.constant"() <{value = 30 : i64}> : () -> i64
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
