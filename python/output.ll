; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__constant_49xi8 = private constant [49 x i8] c"/home/nx/ycy/pb/cherry/tests/llama/tokenizer.bin\00", align 64
@__constant_62xi8 = private constant [62 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/token_embeddings.bin\00", align 64
@__constant_67xi8_0 = private constant [67 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_rms_att_weight.bin\00", align 64
@__constant_55xi8_5 = private constant [55 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_wq.bin\00", align 64
@__constant_55xi8_4 = private constant [55 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_wk.bin\00", align 64
@__constant_55xi8_3 = private constant [55 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_wv.bin\00", align 64
@__constant_55xi8_2 = private constant [55 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_wo.bin\00", align 64
@__constant_67xi8 = private constant [67 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_rms_ffn_weight.bin\00", align 64
@__constant_55xi8_1 = private constant [55 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_w1.bin\00", align 64
@__constant_55xi8_0 = private constant [55 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_w2.bin\00", align 64
@__constant_55xi8 = private constant [55 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/layers_w3.bin\00", align 64
@__constant_60xi8 = private constant [60 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/final_rms_norm.bin\00", align 64
@__constant_57xi8 = private constant [57 x i8] c"/home/nx/ycy/pb/cherry/utils/stories110M/output_wcls.bin\00", align 64
@__constant_12x1024x768xf32 = private constant [12 x [1024 x [768 x float]]] zeroinitializer, align 64
@__constant_1x12x64xf32 = private constant [1 x [12 x [64 x float]]] zeroinitializer, align 64
@__constant_3xi64_1 = private constant [3 x i64] [i64 1, i64 12, i64 64], align 64
@__constant_2xi64 = private constant [2 x i64] [i64 1, i64 768], align 64
@__constant_3xi64_0 = private constant [3 x i64] [i64 1, i64 1, i64 768], align 64
@__constant_3xi64 = private constant [3 x i64] [i64 1, i64 1, i64 64], align 64

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

declare void @free_tokenizer()

declare void @end(i64)

declare void @decode(i64, i64)

declare void @start()

declare { ptr, ptr, i64, [2 x i64], [2 x i64] } @cherry_read_weight_2d_768_32000_f32(ptr, ptr, i64, i64, i64, i64, i64)

declare { ptr, ptr, i64, [1 x i64], [1 x i64] } @cherry_read_weight_1d_768_f32(ptr, ptr, i64, i64, i64, i64)

declare { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_2048_768_f32(ptr, ptr, i64, i64, i64, i64, i64, i64)

declare { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_768_2048_f32(ptr, ptr, i64, i64, i64, i64, i64, i64)

declare { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_768_768_f32(ptr, ptr, i64, i64, i64, i64, i64, i64)

declare { ptr, ptr, i64, [2 x i64], [2 x i64] } @cherry_read_weight_2d_12_768_f32(ptr, ptr, i64, i64, i64, i64, i64)

declare { ptr, ptr, i64, [2 x i64], [2 x i64] } @cherry_read_weight_2d_32000_768_f32(ptr, ptr, i64, i64, i64, i64, i64)

declare void @build_tokenizer(i64, ptr, ptr, i64, i64, i64)

define void @host() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 49) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @__constant_49xi8, i64 mul (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64), i64 49), i1 false)
  call void @build_tokenizer(i64 32000, ptr %1, ptr %6, i64 0, i64 49, i64 1)
  %7 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @cherry_read_weight_2d_32000_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_62xi8, i64 0, i64 62, i64 1, i64 32000, i64 768)
  %8 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @cherry_read_weight_2d_12_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_67xi8_0, i64 0, i64 67, i64 1, i64 12, i64 768)
  %9 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_768_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_55xi8_5, i64 0, i64 55, i64 1, i64 12, i64 768, i64 768)
  %10 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_768_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_55xi8_4, i64 0, i64 55, i64 1, i64 12, i64 768, i64 768)
  %11 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_768_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_55xi8_3, i64 0, i64 55, i64 1, i64 12, i64 768, i64 768)
  %12 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_768_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_55xi8_2, i64 0, i64 55, i64 1, i64 12, i64 768, i64 768)
  %13 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @cherry_read_weight_2d_12_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_67xi8, i64 0, i64 67, i64 1, i64 12, i64 768)
  %14 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_768_2048_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_55xi8_1, i64 0, i64 55, i64 1, i64 12, i64 768, i64 2048)
  %15 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_2048_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_55xi8_0, i64 0, i64 55, i64 1, i64 12, i64 2048, i64 768)
  %16 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @cherry_read_weight_3d_12_768_2048_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_55xi8, i64 0, i64 55, i64 1, i64 12, i64 768, i64 2048)
  %17 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @cherry_read_weight_1d_768_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_60xi8, i64 0, i64 60, i64 1, i64 768)
  %18 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @cherry_read_weight_2d_768_32000_f32(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_57xi8, i64 0, i64 57, i64 1, i64 768, i64 32000)
  call void @start()
  %19 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %20 = ptrtoint ptr %19 to i64
  %21 = add i64 %20, 63
  %22 = urem i64 %21, 64
  %23 = sub i64 %21, %22
  %24 = inttoptr i64 %23 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %24, ptr @__constant_12x1024x768xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9437184), i1 false)
  %25 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %26 = ptrtoint ptr %25 to i64
  %27 = add i64 %26, 63
  %28 = urem i64 %27, 64
  %29 = sub i64 %27, %28
  %30 = inttoptr i64 %29 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %30, ptr @__constant_12x1024x768xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9437184), i1 false)
  br label %31

31:                                               ; preds = %1467, %0
  %32 = phi i64 [ %1468, %1467 ], [ 1, %0 ]
  %33 = phi i64 [ %38, %1467 ], [ 0, %0 ]
  %34 = icmp slt i64 %33, 30
  br i1 %34, label %35, label %1469

35:                                               ; preds = %31
  %36 = phi i64 [ %32, %31 ]
  %37 = phi i64 [ %33, %31 ]
  %38 = add i64 %37, 1
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 1
  %40 = mul i64 %36, 768
  %41 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %42 = ptrtoint ptr %41 to i64
  %43 = add i64 %42, 63
  %44 = urem i64 %43, 64
  %45 = sub i64 %43, %44
  %46 = inttoptr i64 %45 to ptr
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %41, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, ptr %46, 1
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 0, 2
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 1, 3, 0
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 768, 3, 1
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 768, 4, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 1, 4, 1
  %54 = getelementptr float, ptr %39, i64 %40
  call void @llvm.memcpy.p0.p0.i64(ptr %46, ptr %54, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %55 = uitofp i64 %37 to float
  br label %56

56:                                               ; preds = %1316, %35
  %57 = phi i64 [ %1317, %1316 ], [ 0, %35 ]
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %1296, %1316 ], [ %53, %35 ]
  %59 = icmp slt i64 %57, 12
  br i1 %59, label %60, label %1318

60:                                               ; preds = %56
  %61 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %62 = mul i64 %57, 768
  %63 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %64 = ptrtoint ptr %63 to i64
  %65 = add i64 %64, 63
  %66 = urem i64 %65, 64
  %67 = sub i64 %65, %66
  %68 = inttoptr i64 %67 to ptr
  br label %69

69:                                               ; preds = %72, %60
  %70 = phi i64 [ %74, %72 ], [ 0, %60 ]
  %71 = icmp slt i64 %70, 1
  br i1 %71, label %72, label %75

72:                                               ; preds = %69
  %73 = getelementptr float, ptr %68, i64 %70
  store float 0.000000e+00, ptr %73, align 4
  %74 = add i64 %70, 1
  br label %69

75:                                               ; preds = %69
  br label %76

76:                                               ; preds = %94, %75
  %77 = phi i64 [ %95, %94 ], [ 0, %75 ]
  %78 = icmp slt i64 %77, 1
  br i1 %78, label %79, label %96

79:                                               ; preds = %76
  br label %80

80:                                               ; preds = %83, %79
  %81 = phi i64 [ %93, %83 ], [ 0, %79 ]
  %82 = icmp slt i64 %81, 768
  br i1 %82, label %83, label %94

83:                                               ; preds = %80
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %85 = mul i64 %77, 768
  %86 = add i64 %85, %81
  %87 = getelementptr float, ptr %84, i64 %86
  %88 = load float, ptr %87, align 4
  %89 = getelementptr float, ptr %68, i64 %77
  %90 = load float, ptr %89, align 4
  %91 = fmul float %88, %88
  %92 = fadd float %90, %91
  store float %92, ptr %89, align 4
  %93 = add i64 %81, 1
  br label %80

94:                                               ; preds = %80
  %95 = add i64 %77, 1
  br label %76

96:                                               ; preds = %76
  %97 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %98 = ptrtoint ptr %97 to i64
  %99 = add i64 %98, 63
  %100 = urem i64 %99, 64
  %101 = sub i64 %99, %100
  %102 = inttoptr i64 %101 to ptr
  br label %103

103:                                              ; preds = %129, %96
  %104 = phi i64 [ %130, %129 ], [ 0, %96 ]
  %105 = icmp slt i64 %104, 1
  br i1 %105, label %106, label %131

106:                                              ; preds = %103
  br label %107

107:                                              ; preds = %110, %106
  %108 = phi i64 [ %128, %110 ], [ 0, %106 ]
  %109 = icmp slt i64 %108, 768
  br i1 %109, label %110, label %129

110:                                              ; preds = %107
  %111 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %112 = mul i64 %104, 768
  %113 = add i64 %112, %108
  %114 = getelementptr float, ptr %111, i64 %113
  %115 = load float, ptr %114, align 4
  %116 = getelementptr float, ptr %68, i64 %104
  %117 = load float, ptr %116, align 4
  %118 = getelementptr float, ptr %61, i64 %62
  %119 = getelementptr float, ptr %118, i64 %108
  %120 = load float, ptr %119, align 4
  %121 = fdiv float %117, 7.680000e+02
  %122 = fadd float %121, 0x3EE4F8B580000000
  %123 = call float @llvm.sqrt.f32(float %122)
  %124 = fdiv float 1.000000e+00, %123
  %125 = fmul float %115, %124
  %126 = fmul float %125, %120
  %127 = getelementptr float, ptr %102, i64 %113
  store float %126, ptr %127, align 4
  %128 = add i64 %108, 1
  br label %107

129:                                              ; preds = %107
  %130 = add i64 %104, 1
  br label %103

131:                                              ; preds = %103
  %132 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %9, 1
  %133 = mul i64 %57, 589824
  %134 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 1
  %135 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %11, 1
  %136 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %137 = ptrtoint ptr %136 to i64
  %138 = add i64 %137, 63
  %139 = urem i64 %138, 64
  %140 = sub i64 %138, %139
  %141 = inttoptr i64 %140 to ptr
  br label %142

142:                                              ; preds = %154, %131
  %143 = phi i64 [ %155, %154 ], [ 0, %131 ]
  %144 = icmp slt i64 %143, 1
  br i1 %144, label %145, label %156

145:                                              ; preds = %142
  br label %146

146:                                              ; preds = %149, %145
  %147 = phi i64 [ %153, %149 ], [ 0, %145 ]
  %148 = icmp slt i64 %147, 768
  br i1 %148, label %149, label %154

149:                                              ; preds = %146
  %150 = mul i64 %143, 768
  %151 = add i64 %150, %147
  %152 = getelementptr float, ptr %141, i64 %151
  store float 0.000000e+00, ptr %152, align 4
  %153 = add i64 %147, 1
  br label %146

154:                                              ; preds = %146
  %155 = add i64 %143, 1
  br label %142

156:                                              ; preds = %142
  %157 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %158 = ptrtoint ptr %157 to i64
  %159 = add i64 %158, 63
  %160 = urem i64 %159, 64
  %161 = sub i64 %159, %160
  %162 = inttoptr i64 %161 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %162, ptr %141, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %163

163:                                              ; preds = %192, %156
  %164 = phi i64 [ %193, %192 ], [ 0, %156 ]
  %165 = icmp slt i64 %164, 1
  br i1 %165, label %166, label %194

166:                                              ; preds = %163
  br label %167

167:                                              ; preds = %190, %166
  %168 = phi i64 [ %191, %190 ], [ 0, %166 ]
  %169 = icmp slt i64 %168, 768
  br i1 %169, label %170, label %192

170:                                              ; preds = %167
  br label %171

171:                                              ; preds = %174, %170
  %172 = phi i64 [ %189, %174 ], [ 0, %170 ]
  %173 = icmp slt i64 %172, 768
  br i1 %173, label %174, label %190

174:                                              ; preds = %171
  %175 = mul i64 %164, 768
  %176 = add i64 %175, %172
  %177 = getelementptr float, ptr %102, i64 %176
  %178 = load float, ptr %177, align 4
  %179 = getelementptr float, ptr %132, i64 %133
  %180 = mul i64 %172, 768
  %181 = add i64 %180, %168
  %182 = getelementptr float, ptr %179, i64 %181
  %183 = load float, ptr %182, align 4
  %184 = add i64 %175, %168
  %185 = getelementptr float, ptr %162, i64 %184
  %186 = load float, ptr %185, align 4
  %187 = fmul float %178, %183
  %188 = fadd float %186, %187
  store float %188, ptr %185, align 4
  %189 = add i64 %172, 1
  br label %171

190:                                              ; preds = %171
  %191 = add i64 %168, 1
  br label %167

192:                                              ; preds = %167
  %193 = add i64 %164, 1
  br label %163

194:                                              ; preds = %163
  %195 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %196 = ptrtoint ptr %195 to i64
  %197 = add i64 %196, 63
  %198 = urem i64 %197, 64
  %199 = sub i64 %197, %198
  %200 = inttoptr i64 %199 to ptr
  br label %201

201:                                              ; preds = %213, %194
  %202 = phi i64 [ %214, %213 ], [ 0, %194 ]
  %203 = icmp slt i64 %202, 1
  br i1 %203, label %204, label %215

204:                                              ; preds = %201
  br label %205

205:                                              ; preds = %208, %204
  %206 = phi i64 [ %212, %208 ], [ 0, %204 ]
  %207 = icmp slt i64 %206, 768
  br i1 %207, label %208, label %213

208:                                              ; preds = %205
  %209 = mul i64 %202, 768
  %210 = add i64 %209, %206
  %211 = getelementptr float, ptr %200, i64 %210
  store float 0.000000e+00, ptr %211, align 4
  %212 = add i64 %206, 1
  br label %205

213:                                              ; preds = %205
  %214 = add i64 %202, 1
  br label %201

215:                                              ; preds = %201
  %216 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %217 = ptrtoint ptr %216 to i64
  %218 = add i64 %217, 63
  %219 = urem i64 %218, 64
  %220 = sub i64 %218, %219
  %221 = inttoptr i64 %220 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %221, ptr %200, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %222

222:                                              ; preds = %251, %215
  %223 = phi i64 [ %252, %251 ], [ 0, %215 ]
  %224 = icmp slt i64 %223, 1
  br i1 %224, label %225, label %253

225:                                              ; preds = %222
  br label %226

226:                                              ; preds = %249, %225
  %227 = phi i64 [ %250, %249 ], [ 0, %225 ]
  %228 = icmp slt i64 %227, 768
  br i1 %228, label %229, label %251

229:                                              ; preds = %226
  br label %230

230:                                              ; preds = %233, %229
  %231 = phi i64 [ %248, %233 ], [ 0, %229 ]
  %232 = icmp slt i64 %231, 768
  br i1 %232, label %233, label %249

233:                                              ; preds = %230
  %234 = mul i64 %223, 768
  %235 = add i64 %234, %231
  %236 = getelementptr float, ptr %102, i64 %235
  %237 = load float, ptr %236, align 4
  %238 = getelementptr float, ptr %134, i64 %133
  %239 = mul i64 %231, 768
  %240 = add i64 %239, %227
  %241 = getelementptr float, ptr %238, i64 %240
  %242 = load float, ptr %241, align 4
  %243 = add i64 %234, %227
  %244 = getelementptr float, ptr %221, i64 %243
  %245 = load float, ptr %244, align 4
  %246 = fmul float %237, %242
  %247 = fadd float %245, %246
  store float %247, ptr %244, align 4
  %248 = add i64 %231, 1
  br label %230

249:                                              ; preds = %230
  %250 = add i64 %227, 1
  br label %226

251:                                              ; preds = %226
  %252 = add i64 %223, 1
  br label %222

253:                                              ; preds = %222
  %254 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %255 = ptrtoint ptr %254 to i64
  %256 = add i64 %255, 63
  %257 = urem i64 %256, 64
  %258 = sub i64 %256, %257
  %259 = inttoptr i64 %258 to ptr
  br label %260

260:                                              ; preds = %272, %253
  %261 = phi i64 [ %273, %272 ], [ 0, %253 ]
  %262 = icmp slt i64 %261, 1
  br i1 %262, label %263, label %274

263:                                              ; preds = %260
  br label %264

264:                                              ; preds = %267, %263
  %265 = phi i64 [ %271, %267 ], [ 0, %263 ]
  %266 = icmp slt i64 %265, 768
  br i1 %266, label %267, label %272

267:                                              ; preds = %264
  %268 = mul i64 %261, 768
  %269 = add i64 %268, %265
  %270 = getelementptr float, ptr %259, i64 %269
  store float 0.000000e+00, ptr %270, align 4
  %271 = add i64 %265, 1
  br label %264

272:                                              ; preds = %264
  %273 = add i64 %261, 1
  br label %260

274:                                              ; preds = %260
  %275 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %276 = ptrtoint ptr %275 to i64
  %277 = add i64 %276, 63
  %278 = urem i64 %277, 64
  %279 = sub i64 %277, %278
  %280 = inttoptr i64 %279 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %280, ptr %259, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %281

281:                                              ; preds = %310, %274
  %282 = phi i64 [ %311, %310 ], [ 0, %274 ]
  %283 = icmp slt i64 %282, 1
  br i1 %283, label %284, label %312

284:                                              ; preds = %281
  br label %285

285:                                              ; preds = %308, %284
  %286 = phi i64 [ %309, %308 ], [ 0, %284 ]
  %287 = icmp slt i64 %286, 768
  br i1 %287, label %288, label %310

288:                                              ; preds = %285
  br label %289

289:                                              ; preds = %292, %288
  %290 = phi i64 [ %307, %292 ], [ 0, %288 ]
  %291 = icmp slt i64 %290, 768
  br i1 %291, label %292, label %308

292:                                              ; preds = %289
  %293 = mul i64 %282, 768
  %294 = add i64 %293, %290
  %295 = getelementptr float, ptr %102, i64 %294
  %296 = load float, ptr %295, align 4
  %297 = getelementptr float, ptr %135, i64 %133
  %298 = mul i64 %290, 768
  %299 = add i64 %298, %286
  %300 = getelementptr float, ptr %297, i64 %299
  %301 = load float, ptr %300, align 4
  %302 = add i64 %293, %286
  %303 = getelementptr float, ptr %280, i64 %302
  %304 = load float, ptr %303, align 4
  %305 = fmul float %296, %301
  %306 = fadd float %304, %305
  store float %306, ptr %303, align 4
  %307 = add i64 %290, 1
  br label %289

308:                                              ; preds = %289
  %309 = add i64 %286, 1
  br label %285

310:                                              ; preds = %285
  %311 = add i64 %282, 1
  br label %281

312:                                              ; preds = %281
  %313 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %314 = ptrtoint ptr %313 to i64
  %315 = add i64 %314, 63
  %316 = urem i64 %315, 64
  %317 = sub i64 %315, %316
  %318 = inttoptr i64 %317 to ptr
  %319 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %320 = ptrtoint ptr %319 to i64
  %321 = add i64 %320, 63
  %322 = urem i64 %321, 64
  %323 = sub i64 %321, %322
  %324 = inttoptr i64 %323 to ptr
  br label %325

325:                                              ; preds = %328, %312
  %326 = phi i64 [ %338, %328 ], [ 0, %312 ]
  %327 = icmp slt i64 %326, 32
  br i1 %327, label %328, label %339

328:                                              ; preds = %325
  %329 = uitofp i64 %326 to float
  %330 = fmul float %329, -2.000000e+00
  %331 = fdiv float %330, 6.400000e+01
  %332 = call float @llvm.pow.f32(float 1.000000e+04, float %331)
  %333 = fmul float %55, %332
  %334 = call float @llvm.cos.f32(float %333)
  %335 = call float @llvm.sin.f32(float %333)
  %336 = getelementptr float, ptr %318, i64 %326
  store float %334, ptr %336, align 4
  %337 = getelementptr float, ptr %324, i64 %326
  store float %335, ptr %337, align 4
  %338 = add i64 %326, 1
  br label %325

339:                                              ; preds = %325
  %340 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %341 = ptrtoint ptr %340 to i64
  %342 = add i64 %341, 63
  %343 = urem i64 %342, 64
  %344 = sub i64 %342, %343
  %345 = inttoptr i64 %344 to ptr
  %346 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %340, 0
  %347 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %346, ptr %345, 1
  %348 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %347, i64 0, 2
  %349 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %348, i64 1, 3, 0
  %350 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %349, i64 12, 3, 1
  %351 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %350, i64 32, 3, 2
  %352 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %351, i64 1, 3, 3
  %353 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %352, i64 384, 4, 0
  %354 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %353, i64 32, 4, 1
  %355 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %354, i64 1, 4, 2
  %356 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %355, i64 1, 4, 3
  %357 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %358 = ptrtoint ptr %357 to i64
  %359 = add i64 %358, 63
  %360 = urem i64 %359, 64
  %361 = sub i64 %359, %360
  %362 = inttoptr i64 %361 to ptr
  %363 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %357, 0
  %364 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %363, ptr %362, 1
  %365 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %364, i64 0, 2
  %366 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %365, i64 1, 3, 0
  %367 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %366, i64 12, 3, 1
  %368 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %367, i64 32, 3, 2
  %369 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %368, i64 1, 3, 3
  %370 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %369, i64 384, 4, 0
  %371 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %370, i64 32, 4, 1
  %372 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %371, i64 1, 4, 2
  %373 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %372, i64 1, 4, 3
  br label %374

374:                                              ; preds = %424, %339
  %375 = phi i64 [ %425, %424 ], [ 0, %339 ]
  %376 = icmp slt i64 %375, 1
  br i1 %376, label %377, label %426

377:                                              ; preds = %374
  br label %378

378:                                              ; preds = %422, %377
  %379 = phi i64 [ %423, %422 ], [ 0, %377 ]
  %380 = icmp slt i64 %379, 12
  br i1 %380, label %381, label %424

381:                                              ; preds = %378
  br label %382

382:                                              ; preds = %420, %381
  %383 = phi i64 [ %421, %420 ], [ 0, %381 ]
  %384 = icmp slt i64 %383, 32
  br i1 %384, label %385, label %422

385:                                              ; preds = %382
  br label %386

386:                                              ; preds = %389, %385
  %387 = phi i64 [ %419, %389 ], [ 0, %385 ]
  %388 = icmp slt i64 %387, 1
  br i1 %388, label %389, label %420

389:                                              ; preds = %386
  %390 = mul i64 %375, 768
  %391 = mul i64 %379, 64
  %392 = add i64 %390, %391
  %393 = mul i64 %383, 2
  %394 = add i64 %392, %393
  %395 = add i64 %394, %387
  %396 = getelementptr float, ptr %162, i64 %395
  %397 = load float, ptr %396, align 4
  %398 = getelementptr float, ptr %162, i32 1
  %399 = getelementptr float, ptr %398, i64 %395
  %400 = load float, ptr %399, align 4
  %401 = add i64 %383, %387
  %402 = getelementptr float, ptr %318, i64 %401
  %403 = load float, ptr %402, align 4
  %404 = getelementptr float, ptr %324, i64 %401
  %405 = load float, ptr %404, align 4
  %406 = fmul float %397, %403
  %407 = fmul float %400, %405
  %408 = fsub float %406, %407
  %409 = fmul float %400, %403
  %410 = fmul float %397, %405
  %411 = fadd float %409, %410
  %412 = mul i64 %375, 384
  %413 = mul i64 %379, 32
  %414 = add i64 %412, %413
  %415 = add i64 %414, %383
  %416 = add i64 %415, %387
  %417 = getelementptr float, ptr %345, i64 %416
  store float %408, ptr %417, align 4
  %418 = getelementptr float, ptr %362, i64 %416
  store float %411, ptr %418, align 4
  %419 = add i64 %387, 1
  br label %386

420:                                              ; preds = %386
  %421 = add i64 %383, 1
  br label %382

422:                                              ; preds = %382
  %423 = add i64 %379, 1
  br label %378

424:                                              ; preds = %378
  %425 = add i64 %375, 1
  br label %374

426:                                              ; preds = %374
  %427 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %428 = ptrtoint ptr %427 to i64
  %429 = add i64 %428, 63
  %430 = urem i64 %429, 64
  %431 = sub i64 %429, %430
  %432 = inttoptr i64 %431 to ptr
  %433 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %427, 0
  %434 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %433, ptr %432, 1
  %435 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %434, i64 0, 2
  %436 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %435, i64 1, 3, 0
  %437 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %436, i64 768, 4, 0
  %438 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %437, i64 12, 3, 1
  %439 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %438, i64 64, 4, 1
  %440 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %439, i64 32, 3, 2
  %441 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %440, i64 2, 4, 2
  %442 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %441, i64 1, 3, 3
  %443 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %442, i64 1, 4, 3
  %444 = call ptr @llvm.stacksave.p0()
  %445 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %356, ptr %445, align 8
  %446 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %445, 1
  %447 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %443, ptr %447, align 8
  %448 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %447, 1
  %449 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %446, ptr %449, align 8
  %450 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %448, ptr %450, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %449, ptr %450)
  call void @llvm.stackrestore.p0(ptr %444)
  %451 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %452 = ptrtoint ptr %451 to i64
  %453 = add i64 %452, 63
  %454 = urem i64 %453, 64
  %455 = sub i64 %453, %454
  %456 = inttoptr i64 %455 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %456, ptr %432, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %457 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %451, 0
  %458 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %457, ptr %456, 1
  %459 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %458, i64 1, 2
  %460 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %459, i64 1, 3, 0
  %461 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %460, i64 768, 4, 0
  %462 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %461, i64 12, 3, 1
  %463 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %462, i64 64, 4, 1
  %464 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %463, i64 32, 3, 2
  %465 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %464, i64 2, 4, 2
  %466 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %465, i64 1, 3, 3
  %467 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %466, i64 1, 4, 3
  %468 = call ptr @llvm.stacksave.p0()
  %469 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %373, ptr %469, align 8
  %470 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %469, 1
  %471 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %467, ptr %471, align 8
  %472 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %471, 1
  %473 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %470, ptr %473, align 8
  %474 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %472, ptr %474, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %473, ptr %474)
  call void @llvm.stackrestore.p0(ptr %468)
  %475 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %476 = ptrtoint ptr %475 to i64
  %477 = add i64 %476, 63
  %478 = urem i64 %477, 64
  %479 = sub i64 %477, %478
  %480 = inttoptr i64 %479 to ptr
  %481 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %482 = ptrtoint ptr %481 to i64
  %483 = add i64 %482, 63
  %484 = urem i64 %483, 64
  %485 = sub i64 %483, %484
  %486 = inttoptr i64 %485 to ptr
  br label %487

487:                                              ; preds = %490, %426
  %488 = phi i64 [ %500, %490 ], [ 0, %426 ]
  %489 = icmp slt i64 %488, 32
  br i1 %489, label %490, label %501

490:                                              ; preds = %487
  %491 = uitofp i64 %488 to float
  %492 = fmul float %491, -2.000000e+00
  %493 = fdiv float %492, 6.400000e+01
  %494 = call float @llvm.pow.f32(float 1.000000e+04, float %493)
  %495 = fmul float %55, %494
  %496 = call float @llvm.cos.f32(float %495)
  %497 = call float @llvm.sin.f32(float %495)
  %498 = getelementptr float, ptr %480, i64 %488
  store float %496, ptr %498, align 4
  %499 = getelementptr float, ptr %486, i64 %488
  store float %497, ptr %499, align 4
  %500 = add i64 %488, 1
  br label %487

501:                                              ; preds = %487
  %502 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %503 = ptrtoint ptr %502 to i64
  %504 = add i64 %503, 63
  %505 = urem i64 %504, 64
  %506 = sub i64 %504, %505
  %507 = inttoptr i64 %506 to ptr
  %508 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %502, 0
  %509 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %508, ptr %507, 1
  %510 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %509, i64 0, 2
  %511 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %510, i64 1, 3, 0
  %512 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %511, i64 12, 3, 1
  %513 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %512, i64 32, 3, 2
  %514 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %513, i64 1, 3, 3
  %515 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %514, i64 384, 4, 0
  %516 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %515, i64 32, 4, 1
  %517 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %516, i64 1, 4, 2
  %518 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %517, i64 1, 4, 3
  %519 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %520 = ptrtoint ptr %519 to i64
  %521 = add i64 %520, 63
  %522 = urem i64 %521, 64
  %523 = sub i64 %521, %522
  %524 = inttoptr i64 %523 to ptr
  %525 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %519, 0
  %526 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %525, ptr %524, 1
  %527 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %526, i64 0, 2
  %528 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %527, i64 1, 3, 0
  %529 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %528, i64 12, 3, 1
  %530 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %529, i64 32, 3, 2
  %531 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %530, i64 1, 3, 3
  %532 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %531, i64 384, 4, 0
  %533 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %532, i64 32, 4, 1
  %534 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %533, i64 1, 4, 2
  %535 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %534, i64 1, 4, 3
  br label %536

536:                                              ; preds = %586, %501
  %537 = phi i64 [ %587, %586 ], [ 0, %501 ]
  %538 = icmp slt i64 %537, 1
  br i1 %538, label %539, label %588

539:                                              ; preds = %536
  br label %540

540:                                              ; preds = %584, %539
  %541 = phi i64 [ %585, %584 ], [ 0, %539 ]
  %542 = icmp slt i64 %541, 12
  br i1 %542, label %543, label %586

543:                                              ; preds = %540
  br label %544

544:                                              ; preds = %582, %543
  %545 = phi i64 [ %583, %582 ], [ 0, %543 ]
  %546 = icmp slt i64 %545, 32
  br i1 %546, label %547, label %584

547:                                              ; preds = %544
  br label %548

548:                                              ; preds = %551, %547
  %549 = phi i64 [ %581, %551 ], [ 0, %547 ]
  %550 = icmp slt i64 %549, 1
  br i1 %550, label %551, label %582

551:                                              ; preds = %548
  %552 = mul i64 %537, 768
  %553 = mul i64 %541, 64
  %554 = add i64 %552, %553
  %555 = mul i64 %545, 2
  %556 = add i64 %554, %555
  %557 = add i64 %556, %549
  %558 = getelementptr float, ptr %221, i64 %557
  %559 = load float, ptr %558, align 4
  %560 = getelementptr float, ptr %221, i32 1
  %561 = getelementptr float, ptr %560, i64 %557
  %562 = load float, ptr %561, align 4
  %563 = add i64 %545, %549
  %564 = getelementptr float, ptr %480, i64 %563
  %565 = load float, ptr %564, align 4
  %566 = getelementptr float, ptr %486, i64 %563
  %567 = load float, ptr %566, align 4
  %568 = fmul float %559, %565
  %569 = fmul float %562, %567
  %570 = fsub float %568, %569
  %571 = fmul float %562, %565
  %572 = fmul float %559, %567
  %573 = fadd float %571, %572
  %574 = mul i64 %537, 384
  %575 = mul i64 %541, 32
  %576 = add i64 %574, %575
  %577 = add i64 %576, %545
  %578 = add i64 %577, %549
  %579 = getelementptr float, ptr %507, i64 %578
  store float %570, ptr %579, align 4
  %580 = getelementptr float, ptr %524, i64 %578
  store float %573, ptr %580, align 4
  %581 = add i64 %549, 1
  br label %548

582:                                              ; preds = %548
  %583 = add i64 %545, 1
  br label %544

584:                                              ; preds = %544
  %585 = add i64 %541, 1
  br label %540

586:                                              ; preds = %540
  %587 = add i64 %537, 1
  br label %536

588:                                              ; preds = %536
  %589 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %590 = ptrtoint ptr %589 to i64
  %591 = add i64 %590, 63
  %592 = urem i64 %591, 64
  %593 = sub i64 %591, %592
  %594 = inttoptr i64 %593 to ptr
  %595 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %589, 0
  %596 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %595, ptr %594, 1
  %597 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %596, i64 0, 2
  %598 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %597, i64 1, 3, 0
  %599 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %598, i64 768, 4, 0
  %600 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %599, i64 12, 3, 1
  %601 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %600, i64 64, 4, 1
  %602 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %601, i64 32, 3, 2
  %603 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %602, i64 2, 4, 2
  %604 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %603, i64 1, 3, 3
  %605 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %604, i64 1, 4, 3
  %606 = call ptr @llvm.stacksave.p0()
  %607 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %518, ptr %607, align 8
  %608 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %607, 1
  %609 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %605, ptr %609, align 8
  %610 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %609, 1
  %611 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %608, ptr %611, align 8
  %612 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %610, ptr %612, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %611, ptr %612)
  call void @llvm.stackrestore.p0(ptr %606)
  %613 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %614 = ptrtoint ptr %613 to i64
  %615 = add i64 %614, 63
  %616 = urem i64 %615, 64
  %617 = sub i64 %615, %616
  %618 = inttoptr i64 %617 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %618, ptr %594, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %619 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %613, 0
  %620 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %619, ptr %618, 1
  %621 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %620, i64 1, 2
  %622 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %621, i64 1, 3, 0
  %623 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %622, i64 768, 4, 0
  %624 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %623, i64 12, 3, 1
  %625 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %624, i64 64, 4, 1
  %626 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %625, i64 32, 3, 2
  %627 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %626, i64 2, 4, 2
  %628 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %627, i64 1, 3, 3
  %629 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %628, i64 1, 4, 3
  %630 = call ptr @llvm.stacksave.p0()
  %631 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %535, ptr %631, align 8
  %632 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %631, 1
  %633 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %629, ptr %633, align 8
  %634 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %633, 1
  %635 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %632, ptr %635, align 8
  %636 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %634, ptr %636, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %635, ptr %636)
  call void @llvm.stackrestore.p0(ptr %630)
  %637 = mul i64 %57, 786432
  %638 = mul i64 %37, 768
  %639 = add i64 %637, %638
  %640 = getelementptr float, ptr %24, i64 %639
  call void @llvm.memcpy.p0.p0.i64(ptr %640, ptr %618, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %641 = getelementptr float, ptr %30, i64 %639
  call void @llvm.memcpy.p0.p0.i64(ptr %641, ptr %280, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %642 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %643 = ptrtoint ptr %642 to i64
  %644 = add i64 %643, 63
  %645 = urem i64 %644, 64
  %646 = sub i64 %644, %645
  %647 = inttoptr i64 %646 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %647, ptr @__constant_1x12x64xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %648

648:                                              ; preds = %957, %588
  %649 = phi i64 [ %960, %957 ], [ 0, %588 ]
  %650 = icmp slt i64 %649, 12
  br i1 %650, label %651, label %961

651:                                              ; preds = %648
  %652 = mul i64 %649, 64
  %653 = add i64 %637, %652
  %654 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 65536) to i64), i64 64))
  %655 = ptrtoint ptr %654 to i64
  %656 = add i64 %655, 63
  %657 = urem i64 %656, 64
  %658 = sub i64 %656, %657
  %659 = inttoptr i64 %658 to ptr
  br label %660

660:                                              ; preds = %677, %651
  %661 = phi i64 [ %678, %677 ], [ 0, %651 ]
  %662 = icmp slt i64 %661, 64
  br i1 %662, label %663, label %679

663:                                              ; preds = %660
  br label %664

664:                                              ; preds = %667, %663
  %665 = phi i64 [ %676, %667 ], [ 0, %663 ]
  %666 = icmp slt i64 %665, 1024
  br i1 %666, label %667, label %677

667:                                              ; preds = %664
  %668 = getelementptr float, ptr %24, i64 %653
  %669 = mul i64 %665, 768
  %670 = add i64 %669, %661
  %671 = getelementptr float, ptr %668, i64 %670
  %672 = load float, ptr %671, align 4
  %673 = mul i64 %661, 1024
  %674 = add i64 %673, %665
  %675 = getelementptr float, ptr %659, i64 %674
  store float %672, ptr %675, align 4
  %676 = add i64 %665, 1
  br label %664

677:                                              ; preds = %664
  %678 = add i64 %661, 1
  br label %660

679:                                              ; preds = %660
  %680 = mul i64 %38, 1
  %681 = getelementptr float, ptr null, i64 %680
  %682 = ptrtoint ptr %681 to i64
  %683 = add i64 %682, 64
  %684 = call ptr @malloc(i64 %683)
  %685 = ptrtoint ptr %684 to i64
  %686 = add i64 %685, 63
  %687 = urem i64 %686, 64
  %688 = sub i64 %686, %687
  %689 = inttoptr i64 %688 to ptr
  %690 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %684, 0
  %691 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %690, ptr %689, 1
  %692 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %691, i64 0, 2
  %693 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %692, i64 1, 3, 0
  %694 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %693, i64 %38, 3, 1
  %695 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %694, i64 %38, 4, 0
  %696 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %695, i64 1, 4, 1
  br label %697

697:                                              ; preds = %709, %679
  %698 = phi i64 [ %710, %709 ], [ 0, %679 ]
  %699 = icmp slt i64 %698, 1
  br i1 %699, label %700, label %711

700:                                              ; preds = %697
  br label %701

701:                                              ; preds = %704, %700
  %702 = phi i64 [ %708, %704 ], [ 0, %700 ]
  %703 = icmp slt i64 %702, %38
  br i1 %703, label %704, label %709

704:                                              ; preds = %701
  %705 = mul i64 %698, %38
  %706 = add i64 %705, %702
  %707 = getelementptr float, ptr %689, i64 %706
  store float 0.000000e+00, ptr %707, align 4
  %708 = add i64 %702, 1
  br label %701

709:                                              ; preds = %701
  %710 = add i64 %698, 1
  br label %697

711:                                              ; preds = %697
  br label %712

712:                                              ; preds = %742, %711
  %713 = phi i64 [ %743, %742 ], [ 0, %711 ]
  %714 = icmp slt i64 %713, 1
  br i1 %714, label %715, label %744

715:                                              ; preds = %712
  br label %716

716:                                              ; preds = %740, %715
  %717 = phi i64 [ %741, %740 ], [ 0, %715 ]
  %718 = icmp slt i64 %717, %38
  br i1 %718, label %719, label %742

719:                                              ; preds = %716
  br label %720

720:                                              ; preds = %723, %719
  %721 = phi i64 [ %739, %723 ], [ 0, %719 ]
  %722 = icmp slt i64 %721, 64
  br i1 %722, label %723, label %740

723:                                              ; preds = %720
  %724 = getelementptr float, ptr %456, i64 %652
  %725 = mul i64 %713, 768
  %726 = add i64 %725, %721
  %727 = getelementptr float, ptr %724, i64 %726
  %728 = load float, ptr %727, align 4
  %729 = mul i64 %721, 1024
  %730 = add i64 %729, %717
  %731 = getelementptr float, ptr %659, i64 %730
  %732 = load float, ptr %731, align 4
  %733 = mul i64 %713, %38
  %734 = add i64 %733, %717
  %735 = getelementptr float, ptr %689, i64 %734
  %736 = load float, ptr %735, align 4
  %737 = fmul float %728, %732
  %738 = fadd float %736, %737
  store float %738, ptr %735, align 4
  %739 = add i64 %721, 1
  br label %720

740:                                              ; preds = %720
  %741 = add i64 %717, 1
  br label %716

742:                                              ; preds = %716
  %743 = add i64 %713, 1
  br label %712

744:                                              ; preds = %712
  %745 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %746 = ptrtoint ptr %745 to i64
  %747 = add i64 %746, 63
  %748 = urem i64 %747, 64
  %749 = sub i64 %747, %748
  %750 = inttoptr i64 %749 to ptr
  br label %751

751:                                              ; preds = %763, %744
  %752 = phi i64 [ %764, %763 ], [ 0, %744 ]
  %753 = icmp slt i64 %752, 1
  br i1 %753, label %754, label %765

754:                                              ; preds = %751
  br label %755

755:                                              ; preds = %758, %754
  %756 = phi i64 [ %762, %758 ], [ 0, %754 ]
  %757 = icmp slt i64 %756, 1024
  br i1 %757, label %758, label %763

758:                                              ; preds = %755
  %759 = mul i64 %752, 1024
  %760 = add i64 %759, %756
  %761 = getelementptr float, ptr %750, i64 %760
  store float -1.000000e+09, ptr %761, align 4
  %762 = add i64 %756, 1
  br label %755

763:                                              ; preds = %755
  %764 = add i64 %752, 1
  br label %751

765:                                              ; preds = %751
  %766 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %745, 0
  %767 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %766, ptr %750, 1
  %768 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %767, i64 0, 2
  %769 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %768, i64 1, 3, 0
  %770 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %769, i64 1024, 4, 0
  %771 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %770, i64 %38, 3, 1
  %772 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %771, i64 1, 4, 1
  %773 = call ptr @llvm.stacksave.p0()
  %774 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %696, ptr %774, align 8
  %775 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %774, 1
  %776 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %772, ptr %776, align 8
  %777 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %776, 1
  %778 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %775, ptr %778, align 8
  %779 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %777, ptr %779, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %778, ptr %779)
  call void @llvm.stackrestore.p0(ptr %773)
  %780 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %781 = ptrtoint ptr %780 to i64
  %782 = add i64 %781, 63
  %783 = urem i64 %782, 64
  %784 = sub i64 %782, %783
  %785 = inttoptr i64 %784 to ptr
  br label %786

786:                                              ; preds = %801, %765
  %787 = phi i64 [ %802, %801 ], [ 0, %765 ]
  %788 = icmp slt i64 %787, 1
  br i1 %788, label %789, label %803

789:                                              ; preds = %786
  br label %790

790:                                              ; preds = %793, %789
  %791 = phi i64 [ %800, %793 ], [ 0, %789 ]
  %792 = icmp slt i64 %791, 1024
  br i1 %792, label %793, label %801

793:                                              ; preds = %790
  %794 = mul i64 %787, 1024
  %795 = add i64 %794, %791
  %796 = getelementptr float, ptr %750, i64 %795
  %797 = load float, ptr %796, align 4
  %798 = fmul float %797, 1.250000e-01
  %799 = getelementptr float, ptr %785, i64 %795
  store float %798, ptr %799, align 4
  %800 = add i64 %791, 1
  br label %790

801:                                              ; preds = %790
  %802 = add i64 %787, 1
  br label %786

803:                                              ; preds = %786
  %804 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %805 = ptrtoint ptr %804 to i64
  %806 = add i64 %805, 63
  %807 = urem i64 %806, 64
  %808 = sub i64 %806, %807
  %809 = inttoptr i64 %808 to ptr
  br label %810

810:                                              ; preds = %813, %803
  %811 = phi i64 [ %815, %813 ], [ 0, %803 ]
  %812 = icmp slt i64 %811, 1
  br i1 %812, label %813, label %816

813:                                              ; preds = %810
  %814 = getelementptr float, ptr %809, i64 %811
  store float 0xFFF0000000000000, ptr %814, align 4
  %815 = add i64 %811, 1
  br label %810

816:                                              ; preds = %810
  br label %817

817:                                              ; preds = %833, %816
  %818 = phi i64 [ %834, %833 ], [ 0, %816 ]
  %819 = icmp slt i64 %818, 1
  br i1 %819, label %820, label %835

820:                                              ; preds = %817
  br label %821

821:                                              ; preds = %824, %820
  %822 = phi i64 [ %832, %824 ], [ 0, %820 ]
  %823 = icmp slt i64 %822, 1024
  br i1 %823, label %824, label %833

824:                                              ; preds = %821
  %825 = mul i64 %818, 1024
  %826 = add i64 %825, %822
  %827 = getelementptr float, ptr %785, i64 %826
  %828 = load float, ptr %827, align 4
  %829 = getelementptr float, ptr %809, i64 %818
  %830 = load float, ptr %829, align 4
  %831 = call float @llvm.maxnum.f32(float %828, float %830)
  store float %831, ptr %829, align 4
  %832 = add i64 %822, 1
  br label %821

833:                                              ; preds = %821
  %834 = add i64 %818, 1
  br label %817

835:                                              ; preds = %817
  %836 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %837 = ptrtoint ptr %836 to i64
  %838 = add i64 %837, 63
  %839 = urem i64 %838, 64
  %840 = sub i64 %838, %839
  %841 = inttoptr i64 %840 to ptr
  br label %842

842:                                              ; preds = %860, %835
  %843 = phi i64 [ %861, %860 ], [ 0, %835 ]
  %844 = icmp slt i64 %843, 1
  br i1 %844, label %845, label %862

845:                                              ; preds = %842
  br label %846

846:                                              ; preds = %849, %845
  %847 = phi i64 [ %859, %849 ], [ 0, %845 ]
  %848 = icmp slt i64 %847, 1024
  br i1 %848, label %849, label %860

849:                                              ; preds = %846
  %850 = mul i64 %843, 1024
  %851 = add i64 %850, %847
  %852 = getelementptr float, ptr %785, i64 %851
  %853 = load float, ptr %852, align 4
  %854 = getelementptr float, ptr %809, i64 %843
  %855 = load float, ptr %854, align 4
  %856 = fsub float %853, %855
  %857 = call float @llvm.exp.f32(float %856)
  %858 = getelementptr float, ptr %841, i64 %851
  store float %857, ptr %858, align 4
  %859 = add i64 %847, 1
  br label %846

860:                                              ; preds = %846
  %861 = add i64 %843, 1
  br label %842

862:                                              ; preds = %842
  %863 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %864 = ptrtoint ptr %863 to i64
  %865 = add i64 %864, 63
  %866 = urem i64 %865, 64
  %867 = sub i64 %865, %866
  %868 = inttoptr i64 %867 to ptr
  br label %869

869:                                              ; preds = %872, %862
  %870 = phi i64 [ %874, %872 ], [ 0, %862 ]
  %871 = icmp slt i64 %870, 1
  br i1 %871, label %872, label %875

872:                                              ; preds = %869
  %873 = getelementptr float, ptr %868, i64 %870
  store float 0.000000e+00, ptr %873, align 4
  %874 = add i64 %870, 1
  br label %869

875:                                              ; preds = %869
  br label %876

876:                                              ; preds = %892, %875
  %877 = phi i64 [ %893, %892 ], [ 0, %875 ]
  %878 = icmp slt i64 %877, 1
  br i1 %878, label %879, label %894

879:                                              ; preds = %876
  br label %880

880:                                              ; preds = %883, %879
  %881 = phi i64 [ %891, %883 ], [ 0, %879 ]
  %882 = icmp slt i64 %881, 1024
  br i1 %882, label %883, label %892

883:                                              ; preds = %880
  %884 = mul i64 %877, 1024
  %885 = add i64 %884, %881
  %886 = getelementptr float, ptr %841, i64 %885
  %887 = load float, ptr %886, align 4
  %888 = getelementptr float, ptr %868, i64 %877
  %889 = load float, ptr %888, align 4
  %890 = fadd float %887, %889
  store float %890, ptr %888, align 4
  %891 = add i64 %881, 1
  br label %880

892:                                              ; preds = %880
  %893 = add i64 %877, 1
  br label %876

894:                                              ; preds = %876
  %895 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64), i64 64))
  %896 = ptrtoint ptr %895 to i64
  %897 = add i64 %896, 63
  %898 = urem i64 %897, 64
  %899 = sub i64 %897, %898
  %900 = inttoptr i64 %899 to ptr
  br label %901

901:                                              ; preds = %913, %894
  %902 = phi i64 [ %914, %913 ], [ 0, %894 ]
  %903 = icmp slt i64 %902, 1
  br i1 %903, label %904, label %915

904:                                              ; preds = %901
  br label %905

905:                                              ; preds = %908, %904
  %906 = phi i64 [ %912, %908 ], [ 0, %904 ]
  %907 = icmp slt i64 %906, 64
  br i1 %907, label %908, label %913

908:                                              ; preds = %905
  %909 = mul i64 %902, 64
  %910 = add i64 %909, %906
  %911 = getelementptr float, ptr %900, i64 %910
  store float 0.000000e+00, ptr %911, align 4
  %912 = add i64 %906, 1
  br label %905

913:                                              ; preds = %905
  %914 = add i64 %902, 1
  br label %901

915:                                              ; preds = %901
  %916 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64), i64 64))
  %917 = ptrtoint ptr %916 to i64
  %918 = add i64 %917, 63
  %919 = urem i64 %918, 64
  %920 = sub i64 %918, %919
  %921 = inttoptr i64 %920 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %921, ptr %900, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64), i1 false)
  br label %922

922:                                              ; preds = %955, %915
  %923 = phi i64 [ %956, %955 ], [ 0, %915 ]
  %924 = icmp slt i64 %923, 1
  br i1 %924, label %925, label %957

925:                                              ; preds = %922
  br label %926

926:                                              ; preds = %953, %925
  %927 = phi i64 [ %954, %953 ], [ 0, %925 ]
  %928 = icmp slt i64 %927, 64
  br i1 %928, label %929, label %955

929:                                              ; preds = %926
  br label %930

930:                                              ; preds = %933, %929
  %931 = phi i64 [ %952, %933 ], [ 0, %929 ]
  %932 = icmp slt i64 %931, 1024
  br i1 %932, label %933, label %953

933:                                              ; preds = %930
  %934 = mul i64 %923, 1024
  %935 = add i64 %934, %931
  %936 = getelementptr float, ptr %841, i64 %935
  %937 = load float, ptr %936, align 4
  %938 = getelementptr float, ptr %868, i64 %923
  %939 = load float, ptr %938, align 4
  %940 = getelementptr float, ptr %30, i64 %653
  %941 = mul i64 %931, 768
  %942 = add i64 %941, %927
  %943 = getelementptr float, ptr %940, i64 %942
  %944 = load float, ptr %943, align 4
  %945 = mul i64 %923, 64
  %946 = add i64 %945, %927
  %947 = getelementptr float, ptr %921, i64 %946
  %948 = load float, ptr %947, align 4
  %949 = fdiv float %937, %939
  %950 = fmul float %949, %944
  %951 = fadd float %948, %950
  store float %951, ptr %947, align 4
  %952 = add i64 %931, 1
  br label %930

953:                                              ; preds = %930
  %954 = add i64 %927, 1
  br label %926

955:                                              ; preds = %926
  %956 = add i64 %923, 1
  br label %922

957:                                              ; preds = %922
  %958 = mul i64 %649, 64
  %959 = getelementptr float, ptr %647, i64 %958
  call void @llvm.memcpy.p0.p0.i64(ptr %959, ptr %921, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64), i1 false)
  %960 = add i64 %649, 1
  br label %648

961:                                              ; preds = %648
  %962 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %12, 1
  %963 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %964 = ptrtoint ptr %963 to i64
  %965 = add i64 %964, 63
  %966 = urem i64 %965, 64
  %967 = sub i64 %965, %966
  %968 = inttoptr i64 %967 to ptr
  br label %969

969:                                              ; preds = %981, %961
  %970 = phi i64 [ %982, %981 ], [ 0, %961 ]
  %971 = icmp slt i64 %970, 1
  br i1 %971, label %972, label %983

972:                                              ; preds = %969
  br label %973

973:                                              ; preds = %976, %972
  %974 = phi i64 [ %980, %976 ], [ 0, %972 ]
  %975 = icmp slt i64 %974, 768
  br i1 %975, label %976, label %981

976:                                              ; preds = %973
  %977 = mul i64 %970, 768
  %978 = add i64 %977, %974
  %979 = getelementptr float, ptr %968, i64 %978
  store float 0.000000e+00, ptr %979, align 4
  %980 = add i64 %974, 1
  br label %973

981:                                              ; preds = %973
  %982 = add i64 %970, 1
  br label %969

983:                                              ; preds = %969
  br label %984

984:                                              ; preds = %1013, %983
  %985 = phi i64 [ %1014, %1013 ], [ 0, %983 ]
  %986 = icmp slt i64 %985, 1
  br i1 %986, label %987, label %1015

987:                                              ; preds = %984
  br label %988

988:                                              ; preds = %1011, %987
  %989 = phi i64 [ %1012, %1011 ], [ 0, %987 ]
  %990 = icmp slt i64 %989, 768
  br i1 %990, label %991, label %1013

991:                                              ; preds = %988
  br label %992

992:                                              ; preds = %995, %991
  %993 = phi i64 [ %1010, %995 ], [ 0, %991 ]
  %994 = icmp slt i64 %993, 768
  br i1 %994, label %995, label %1011

995:                                              ; preds = %992
  %996 = mul i64 %985, 768
  %997 = add i64 %996, %993
  %998 = getelementptr float, ptr %647, i64 %997
  %999 = load float, ptr %998, align 4
  %1000 = getelementptr float, ptr %962, i64 %133
  %1001 = mul i64 %993, 768
  %1002 = add i64 %1001, %989
  %1003 = getelementptr float, ptr %1000, i64 %1002
  %1004 = load float, ptr %1003, align 4
  %1005 = add i64 %996, %989
  %1006 = getelementptr float, ptr %968, i64 %1005
  %1007 = load float, ptr %1006, align 4
  %1008 = fmul float %999, %1004
  %1009 = fadd float %1007, %1008
  store float %1009, ptr %1006, align 4
  %1010 = add i64 %993, 1
  br label %992

1011:                                             ; preds = %992
  %1012 = add i64 %989, 1
  br label %988

1013:                                             ; preds = %988
  %1014 = add i64 %985, 1
  br label %984

1015:                                             ; preds = %984
  %1016 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1017 = ptrtoint ptr %1016 to i64
  %1018 = add i64 %1017, 63
  %1019 = urem i64 %1018, 64
  %1020 = sub i64 %1018, %1019
  %1021 = inttoptr i64 %1020 to ptr
  br label %1022

1022:                                             ; preds = %1040, %1015
  %1023 = phi i64 [ %1041, %1040 ], [ 0, %1015 ]
  %1024 = icmp slt i64 %1023, 1
  br i1 %1024, label %1025, label %1042

1025:                                             ; preds = %1022
  br label %1026

1026:                                             ; preds = %1029, %1025
  %1027 = phi i64 [ %1039, %1029 ], [ 0, %1025 ]
  %1028 = icmp slt i64 %1027, 768
  br i1 %1028, label %1029, label %1040

1029:                                             ; preds = %1026
  %1030 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %1031 = mul i64 %1023, 768
  %1032 = add i64 %1031, %1027
  %1033 = getelementptr float, ptr %1030, i64 %1032
  %1034 = load float, ptr %1033, align 4
  %1035 = getelementptr float, ptr %968, i64 %1032
  %1036 = load float, ptr %1035, align 4
  %1037 = fadd float %1034, %1036
  %1038 = getelementptr float, ptr %1021, i64 %1032
  store float %1037, ptr %1038, align 4
  %1039 = add i64 %1027, 1
  br label %1026

1040:                                             ; preds = %1026
  %1041 = add i64 %1023, 1
  br label %1022

1042:                                             ; preds = %1022
  %1043 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  %1044 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1045 = ptrtoint ptr %1044 to i64
  %1046 = add i64 %1045, 63
  %1047 = urem i64 %1046, 64
  %1048 = sub i64 %1046, %1047
  %1049 = inttoptr i64 %1048 to ptr
  br label %1050

1050:                                             ; preds = %1053, %1042
  %1051 = phi i64 [ %1055, %1053 ], [ 0, %1042 ]
  %1052 = icmp slt i64 %1051, 1
  br i1 %1052, label %1053, label %1056

1053:                                             ; preds = %1050
  %1054 = getelementptr float, ptr %1049, i64 %1051
  store float 0.000000e+00, ptr %1054, align 4
  %1055 = add i64 %1051, 1
  br label %1050

1056:                                             ; preds = %1050
  br label %1057

1057:                                             ; preds = %1074, %1056
  %1058 = phi i64 [ %1075, %1074 ], [ 0, %1056 ]
  %1059 = icmp slt i64 %1058, 1
  br i1 %1059, label %1060, label %1076

1060:                                             ; preds = %1057
  br label %1061

1061:                                             ; preds = %1064, %1060
  %1062 = phi i64 [ %1073, %1064 ], [ 0, %1060 ]
  %1063 = icmp slt i64 %1062, 768
  br i1 %1063, label %1064, label %1074

1064:                                             ; preds = %1061
  %1065 = mul i64 %1058, 768
  %1066 = add i64 %1065, %1062
  %1067 = getelementptr float, ptr %1021, i64 %1066
  %1068 = load float, ptr %1067, align 4
  %1069 = getelementptr float, ptr %1049, i64 %1058
  %1070 = load float, ptr %1069, align 4
  %1071 = fmul float %1068, %1068
  %1072 = fadd float %1070, %1071
  store float %1072, ptr %1069, align 4
  %1073 = add i64 %1062, 1
  br label %1061

1074:                                             ; preds = %1061
  %1075 = add i64 %1058, 1
  br label %1057

1076:                                             ; preds = %1057
  %1077 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1078 = ptrtoint ptr %1077 to i64
  %1079 = add i64 %1078, 63
  %1080 = urem i64 %1079, 64
  %1081 = sub i64 %1079, %1080
  %1082 = inttoptr i64 %1081 to ptr
  br label %1083

1083:                                             ; preds = %1108, %1076
  %1084 = phi i64 [ %1109, %1108 ], [ 0, %1076 ]
  %1085 = icmp slt i64 %1084, 1
  br i1 %1085, label %1086, label %1110

1086:                                             ; preds = %1083
  br label %1087

1087:                                             ; preds = %1090, %1086
  %1088 = phi i64 [ %1107, %1090 ], [ 0, %1086 ]
  %1089 = icmp slt i64 %1088, 768
  br i1 %1089, label %1090, label %1108

1090:                                             ; preds = %1087
  %1091 = mul i64 %1084, 768
  %1092 = add i64 %1091, %1088
  %1093 = getelementptr float, ptr %1021, i64 %1092
  %1094 = load float, ptr %1093, align 4
  %1095 = getelementptr float, ptr %1049, i64 %1084
  %1096 = load float, ptr %1095, align 4
  %1097 = getelementptr float, ptr %1043, i64 %62
  %1098 = getelementptr float, ptr %1097, i64 %1088
  %1099 = load float, ptr %1098, align 4
  %1100 = fdiv float %1096, 7.680000e+02
  %1101 = fadd float %1100, 0x3EE4F8B580000000
  %1102 = call float @llvm.sqrt.f32(float %1101)
  %1103 = fdiv float 1.000000e+00, %1102
  %1104 = fmul float %1094, %1103
  %1105 = fmul float %1104, %1099
  %1106 = getelementptr float, ptr %1082, i64 %1092
  store float %1105, ptr %1106, align 4
  %1107 = add i64 %1088, 1
  br label %1087

1108:                                             ; preds = %1087
  %1109 = add i64 %1084, 1
  br label %1083

1110:                                             ; preds = %1083
  %1111 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 1
  %1112 = mul i64 %57, 1572864
  %1113 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %16, 1
  %1114 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %1115 = ptrtoint ptr %1114 to i64
  %1116 = add i64 %1115, 63
  %1117 = urem i64 %1116, 64
  %1118 = sub i64 %1116, %1117
  %1119 = inttoptr i64 %1118 to ptr
  br label %1120

1120:                                             ; preds = %1132, %1110
  %1121 = phi i64 [ %1133, %1132 ], [ 0, %1110 ]
  %1122 = icmp slt i64 %1121, 1
  br i1 %1122, label %1123, label %1134

1123:                                             ; preds = %1120
  br label %1124

1124:                                             ; preds = %1127, %1123
  %1125 = phi i64 [ %1131, %1127 ], [ 0, %1123 ]
  %1126 = icmp slt i64 %1125, 2048
  br i1 %1126, label %1127, label %1132

1127:                                             ; preds = %1124
  %1128 = mul i64 %1121, 2048
  %1129 = add i64 %1128, %1125
  %1130 = getelementptr float, ptr %1119, i64 %1129
  store float 0.000000e+00, ptr %1130, align 4
  %1131 = add i64 %1125, 1
  br label %1124

1132:                                             ; preds = %1124
  %1133 = add i64 %1121, 1
  br label %1120

1134:                                             ; preds = %1120
  br label %1135

1135:                                             ; preds = %1165, %1134
  %1136 = phi i64 [ %1166, %1165 ], [ 0, %1134 ]
  %1137 = icmp slt i64 %1136, 1
  br i1 %1137, label %1138, label %1167

1138:                                             ; preds = %1135
  br label %1139

1139:                                             ; preds = %1163, %1138
  %1140 = phi i64 [ %1164, %1163 ], [ 0, %1138 ]
  %1141 = icmp slt i64 %1140, 2048
  br i1 %1141, label %1142, label %1165

1142:                                             ; preds = %1139
  br label %1143

1143:                                             ; preds = %1146, %1142
  %1144 = phi i64 [ %1162, %1146 ], [ 0, %1142 ]
  %1145 = icmp slt i64 %1144, 768
  br i1 %1145, label %1146, label %1163

1146:                                             ; preds = %1143
  %1147 = mul i64 %1136, 768
  %1148 = add i64 %1147, %1144
  %1149 = getelementptr float, ptr %1082, i64 %1148
  %1150 = load float, ptr %1149, align 4
  %1151 = getelementptr float, ptr %1111, i64 %1112
  %1152 = mul i64 %1144, 2048
  %1153 = add i64 %1152, %1140
  %1154 = getelementptr float, ptr %1151, i64 %1153
  %1155 = load float, ptr %1154, align 4
  %1156 = mul i64 %1136, 2048
  %1157 = add i64 %1156, %1140
  %1158 = getelementptr float, ptr %1119, i64 %1157
  %1159 = load float, ptr %1158, align 4
  %1160 = fmul float %1150, %1155
  %1161 = fadd float %1159, %1160
  store float %1161, ptr %1158, align 4
  %1162 = add i64 %1144, 1
  br label %1143

1163:                                             ; preds = %1143
  %1164 = add i64 %1140, 1
  br label %1139

1165:                                             ; preds = %1139
  %1166 = add i64 %1136, 1
  br label %1135

1167:                                             ; preds = %1135
  %1168 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %1169 = ptrtoint ptr %1168 to i64
  %1170 = add i64 %1169, 63
  %1171 = urem i64 %1170, 64
  %1172 = sub i64 %1170, %1171
  %1173 = inttoptr i64 %1172 to ptr
  br label %1174

1174:                                             ; preds = %1186, %1167
  %1175 = phi i64 [ %1187, %1186 ], [ 0, %1167 ]
  %1176 = icmp slt i64 %1175, 1
  br i1 %1176, label %1177, label %1188

1177:                                             ; preds = %1174
  br label %1178

1178:                                             ; preds = %1181, %1177
  %1179 = phi i64 [ %1185, %1181 ], [ 0, %1177 ]
  %1180 = icmp slt i64 %1179, 2048
  br i1 %1180, label %1181, label %1186

1181:                                             ; preds = %1178
  %1182 = mul i64 %1175, 2048
  %1183 = add i64 %1182, %1179
  %1184 = getelementptr float, ptr %1173, i64 %1183
  store float 0.000000e+00, ptr %1184, align 4
  %1185 = add i64 %1179, 1
  br label %1178

1186:                                             ; preds = %1178
  %1187 = add i64 %1175, 1
  br label %1174

1188:                                             ; preds = %1174
  br label %1189

1189:                                             ; preds = %1219, %1188
  %1190 = phi i64 [ %1220, %1219 ], [ 0, %1188 ]
  %1191 = icmp slt i64 %1190, 1
  br i1 %1191, label %1192, label %1221

1192:                                             ; preds = %1189
  br label %1193

1193:                                             ; preds = %1217, %1192
  %1194 = phi i64 [ %1218, %1217 ], [ 0, %1192 ]
  %1195 = icmp slt i64 %1194, 2048
  br i1 %1195, label %1196, label %1219

1196:                                             ; preds = %1193
  br label %1197

1197:                                             ; preds = %1200, %1196
  %1198 = phi i64 [ %1216, %1200 ], [ 0, %1196 ]
  %1199 = icmp slt i64 %1198, 768
  br i1 %1199, label %1200, label %1217

1200:                                             ; preds = %1197
  %1201 = mul i64 %1190, 768
  %1202 = add i64 %1201, %1198
  %1203 = getelementptr float, ptr %1082, i64 %1202
  %1204 = load float, ptr %1203, align 4
  %1205 = getelementptr float, ptr %1113, i64 %1112
  %1206 = mul i64 %1198, 2048
  %1207 = add i64 %1206, %1194
  %1208 = getelementptr float, ptr %1205, i64 %1207
  %1209 = load float, ptr %1208, align 4
  %1210 = mul i64 %1190, 2048
  %1211 = add i64 %1210, %1194
  %1212 = getelementptr float, ptr %1173, i64 %1211
  %1213 = load float, ptr %1212, align 4
  %1214 = fmul float %1204, %1209
  %1215 = fadd float %1213, %1214
  store float %1215, ptr %1212, align 4
  %1216 = add i64 %1198, 1
  br label %1197

1217:                                             ; preds = %1197
  %1218 = add i64 %1194, 1
  br label %1193

1219:                                             ; preds = %1193
  %1220 = add i64 %1190, 1
  br label %1189

1221:                                             ; preds = %1189
  %1222 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 1
  %1223 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1224 = ptrtoint ptr %1223 to i64
  %1225 = add i64 %1224, 63
  %1226 = urem i64 %1225, 64
  %1227 = sub i64 %1225, %1226
  %1228 = inttoptr i64 %1227 to ptr
  br label %1229

1229:                                             ; preds = %1241, %1221
  %1230 = phi i64 [ %1242, %1241 ], [ 0, %1221 ]
  %1231 = icmp slt i64 %1230, 1
  br i1 %1231, label %1232, label %1243

1232:                                             ; preds = %1229
  br label %1233

1233:                                             ; preds = %1236, %1232
  %1234 = phi i64 [ %1240, %1236 ], [ 0, %1232 ]
  %1235 = icmp slt i64 %1234, 768
  br i1 %1235, label %1236, label %1241

1236:                                             ; preds = %1233
  %1237 = mul i64 %1230, 768
  %1238 = add i64 %1237, %1234
  %1239 = getelementptr float, ptr %1228, i64 %1238
  store float 0.000000e+00, ptr %1239, align 4
  %1240 = add i64 %1234, 1
  br label %1233

1241:                                             ; preds = %1233
  %1242 = add i64 %1230, 1
  br label %1229

1243:                                             ; preds = %1229
  br label %1244

1244:                                             ; preds = %1281, %1243
  %1245 = phi i64 [ %1282, %1281 ], [ 0, %1243 ]
  %1246 = icmp slt i64 %1245, 1
  br i1 %1246, label %1247, label %1283

1247:                                             ; preds = %1244
  br label %1248

1248:                                             ; preds = %1279, %1247
  %1249 = phi i64 [ %1280, %1279 ], [ 0, %1247 ]
  %1250 = icmp slt i64 %1249, 768
  br i1 %1250, label %1251, label %1281

1251:                                             ; preds = %1248
  br label %1252

1252:                                             ; preds = %1255, %1251
  %1253 = phi i64 [ %1278, %1255 ], [ 0, %1251 ]
  %1254 = icmp slt i64 %1253, 2048
  br i1 %1254, label %1255, label %1279

1255:                                             ; preds = %1252
  %1256 = mul i64 %1245, 2048
  %1257 = add i64 %1256, %1253
  %1258 = getelementptr float, ptr %1119, i64 %1257
  %1259 = load float, ptr %1258, align 4
  %1260 = getelementptr float, ptr %1173, i64 %1257
  %1261 = load float, ptr %1260, align 4
  %1262 = getelementptr float, ptr %1222, i64 %1112
  %1263 = mul i64 %1253, 768
  %1264 = add i64 %1263, %1249
  %1265 = getelementptr float, ptr %1262, i64 %1264
  %1266 = load float, ptr %1265, align 4
  %1267 = mul i64 %1245, 768
  %1268 = add i64 %1267, %1249
  %1269 = getelementptr float, ptr %1228, i64 %1268
  %1270 = load float, ptr %1269, align 4
  %1271 = fneg float %1259
  %1272 = call float @llvm.exp.f32(float %1271)
  %1273 = fadd float %1272, 1.000000e+00
  %1274 = fdiv float %1259, %1273
  %1275 = fmul float %1274, %1261
  %1276 = fmul float %1275, %1266
  %1277 = fadd float %1270, %1276
  store float %1277, ptr %1269, align 4
  %1278 = add i64 %1253, 1
  br label %1252

1279:                                             ; preds = %1252
  %1280 = add i64 %1249, 1
  br label %1248

1281:                                             ; preds = %1248
  %1282 = add i64 %1245, 1
  br label %1244

1283:                                             ; preds = %1244
  %1284 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1285 = ptrtoint ptr %1284 to i64
  %1286 = add i64 %1285, 63
  %1287 = urem i64 %1286, 64
  %1288 = sub i64 %1286, %1287
  %1289 = inttoptr i64 %1288 to ptr
  %1290 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1284, 0
  %1291 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1290, ptr %1289, 1
  %1292 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1291, i64 0, 2
  %1293 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1292, i64 1, 3, 0
  %1294 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1293, i64 768, 3, 1
  %1295 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1294, i64 768, 4, 0
  %1296 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1295, i64 1, 4, 1
  br label %1297

1297:                                             ; preds = %1314, %1283
  %1298 = phi i64 [ %1315, %1314 ], [ 0, %1283 ]
  %1299 = icmp slt i64 %1298, 1
  br i1 %1299, label %1300, label %1316

1300:                                             ; preds = %1297
  br label %1301

1301:                                             ; preds = %1304, %1300
  %1302 = phi i64 [ %1313, %1304 ], [ 0, %1300 ]
  %1303 = icmp slt i64 %1302, 768
  br i1 %1303, label %1304, label %1314

1304:                                             ; preds = %1301
  %1305 = mul i64 %1298, 768
  %1306 = add i64 %1305, %1302
  %1307 = getelementptr float, ptr %1021, i64 %1306
  %1308 = load float, ptr %1307, align 4
  %1309 = getelementptr float, ptr %1228, i64 %1306
  %1310 = load float, ptr %1309, align 4
  %1311 = fadd float %1308, %1310
  %1312 = getelementptr float, ptr %1289, i64 %1306
  store float %1311, ptr %1312, align 4
  %1313 = add i64 %1302, 1
  br label %1301

1314:                                             ; preds = %1301
  %1315 = add i64 %1298, 1
  br label %1297

1316:                                             ; preds = %1297
  %1317 = add i64 %57, 1
  br label %56

1318:                                             ; preds = %56
  %1319 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1320 = ptrtoint ptr %1319 to i64
  %1321 = add i64 %1320, 63
  %1322 = urem i64 %1321, 64
  %1323 = sub i64 %1321, %1322
  %1324 = inttoptr i64 %1323 to ptr
  br label %1325

1325:                                             ; preds = %1328, %1318
  %1326 = phi i64 [ %1330, %1328 ], [ 0, %1318 ]
  %1327 = icmp slt i64 %1326, 1
  br i1 %1327, label %1328, label %1331

1328:                                             ; preds = %1325
  %1329 = getelementptr float, ptr %1324, i64 %1326
  store float 0.000000e+00, ptr %1329, align 4
  %1330 = add i64 %1326, 1
  br label %1325

1331:                                             ; preds = %1325
  br label %1332

1332:                                             ; preds = %1350, %1331
  %1333 = phi i64 [ %1351, %1350 ], [ 0, %1331 ]
  %1334 = icmp slt i64 %1333, 1
  br i1 %1334, label %1335, label %1352

1335:                                             ; preds = %1332
  br label %1336

1336:                                             ; preds = %1339, %1335
  %1337 = phi i64 [ %1349, %1339 ], [ 0, %1335 ]
  %1338 = icmp slt i64 %1337, 768
  br i1 %1338, label %1339, label %1350

1339:                                             ; preds = %1336
  %1340 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %1341 = mul i64 %1333, 768
  %1342 = add i64 %1341, %1337
  %1343 = getelementptr float, ptr %1340, i64 %1342
  %1344 = load float, ptr %1343, align 4
  %1345 = getelementptr float, ptr %1324, i64 %1333
  %1346 = load float, ptr %1345, align 4
  %1347 = fmul float %1344, %1344
  %1348 = fadd float %1346, %1347
  store float %1348, ptr %1345, align 4
  %1349 = add i64 %1337, 1
  br label %1336

1350:                                             ; preds = %1336
  %1351 = add i64 %1333, 1
  br label %1332

1352:                                             ; preds = %1332
  %1353 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32000) to i64), i64 64))
  %1354 = ptrtoint ptr %1353 to i64
  %1355 = add i64 %1354, 63
  %1356 = urem i64 %1355, 64
  %1357 = sub i64 %1355, %1356
  %1358 = inttoptr i64 %1357 to ptr
  br label %1359

1359:                                             ; preds = %1371, %1352
  %1360 = phi i64 [ %1372, %1371 ], [ 0, %1352 ]
  %1361 = icmp slt i64 %1360, 1
  br i1 %1361, label %1362, label %1373

1362:                                             ; preds = %1359
  br label %1363

1363:                                             ; preds = %1366, %1362
  %1364 = phi i64 [ %1370, %1366 ], [ 0, %1362 ]
  %1365 = icmp slt i64 %1364, 32000
  br i1 %1365, label %1366, label %1371

1366:                                             ; preds = %1363
  %1367 = mul i64 %1360, 32000
  %1368 = add i64 %1367, %1364
  %1369 = getelementptr float, ptr %1358, i64 %1368
  store float 0.000000e+00, ptr %1369, align 4
  %1370 = add i64 %1364, 1
  br label %1363

1371:                                             ; preds = %1363
  %1372 = add i64 %1360, 1
  br label %1359

1373:                                             ; preds = %1359
  br label %1374

1374:                                             ; preds = %1416, %1373
  %1375 = phi i64 [ %1417, %1416 ], [ 0, %1373 ]
  %1376 = icmp slt i64 %1375, 1
  br i1 %1376, label %1377, label %1418

1377:                                             ; preds = %1374
  br label %1378

1378:                                             ; preds = %1414, %1377
  %1379 = phi i64 [ %1415, %1414 ], [ 0, %1377 ]
  %1380 = icmp slt i64 %1379, 32000
  br i1 %1380, label %1381, label %1416

1381:                                             ; preds = %1378
  br label %1382

1382:                                             ; preds = %1385, %1381
  %1383 = phi i64 [ %1413, %1385 ], [ 0, %1381 ]
  %1384 = icmp slt i64 %1383, 768
  br i1 %1384, label %1385, label %1414

1385:                                             ; preds = %1382
  %1386 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %1387 = mul i64 %1375, 768
  %1388 = add i64 %1387, %1383
  %1389 = getelementptr float, ptr %1386, i64 %1388
  %1390 = load float, ptr %1389, align 4
  %1391 = getelementptr float, ptr %1324, i64 %1375
  %1392 = load float, ptr %1391, align 4
  %1393 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 1
  %1394 = getelementptr float, ptr %1393, i64 %1383
  %1395 = load float, ptr %1394, align 4
  %1396 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %1397 = mul i64 %1383, 32000
  %1398 = add i64 %1397, %1379
  %1399 = getelementptr float, ptr %1396, i64 %1398
  %1400 = load float, ptr %1399, align 4
  %1401 = mul i64 %1375, 32000
  %1402 = add i64 %1401, %1379
  %1403 = getelementptr float, ptr %1358, i64 %1402
  %1404 = load float, ptr %1403, align 4
  %1405 = fdiv float %1392, 7.680000e+02
  %1406 = fadd float %1405, 0x3EE4F8B580000000
  %1407 = call float @llvm.sqrt.f32(float %1406)
  %1408 = fdiv float 1.000000e+00, %1407
  %1409 = fmul float %1390, %1408
  %1410 = fmul float %1409, %1395
  %1411 = fmul float %1410, %1400
  %1412 = fadd float %1404, %1411
  store float %1412, ptr %1403, align 4
  %1413 = add i64 %1383, 1
  br label %1382

1414:                                             ; preds = %1382
  %1415 = add i64 %1379, 1
  br label %1378

1416:                                             ; preds = %1378
  %1417 = add i64 %1375, 1
  br label %1374

1418:                                             ; preds = %1374
  %1419 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1420 = ptrtoint ptr %1419 to i64
  %1421 = add i64 %1420, 63
  %1422 = urem i64 %1421, 64
  %1423 = sub i64 %1421, %1422
  %1424 = inttoptr i64 %1423 to ptr
  br label %1425

1425:                                             ; preds = %1428, %1418
  %1426 = phi i64 [ %1430, %1428 ], [ 0, %1418 ]
  %1427 = icmp slt i64 %1426, 1
  br i1 %1427, label %1428, label %1431

1428:                                             ; preds = %1425
  %1429 = getelementptr float, ptr %1424, i64 %1426
  store float 0xFFF0000000000000, ptr %1429, align 4
  %1430 = add i64 %1426, 1
  br label %1425

1431:                                             ; preds = %1425
  %1432 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), i64 64))
  %1433 = ptrtoint ptr %1432 to i64
  %1434 = add i64 %1433, 63
  %1435 = urem i64 %1434, 64
  %1436 = sub i64 %1434, %1435
  %1437 = inttoptr i64 %1436 to ptr
  br label %1438

1438:                                             ; preds = %1441, %1431
  %1439 = phi i64 [ %1443, %1441 ], [ 0, %1431 ]
  %1440 = icmp slt i64 %1439, 1
  br i1 %1440, label %1441, label %1444

1441:                                             ; preds = %1438
  %1442 = getelementptr i64, ptr %1437, i64 %1439
  store i64 0, ptr %1442, align 4
  %1443 = add i64 %1439, 1
  br label %1438

1444:                                             ; preds = %1438
  br label %1445

1445:                                             ; preds = %1465, %1444
  %1446 = phi i64 [ %1466, %1465 ], [ 0, %1444 ]
  %1447 = icmp slt i64 %1446, 1
  br i1 %1447, label %1448, label %1467

1448:                                             ; preds = %1445
  br label %1449

1449:                                             ; preds = %1452, %1448
  %1450 = phi i64 [ %1464, %1452 ], [ 0, %1448 ]
  %1451 = icmp slt i64 %1450, 32000
  br i1 %1451, label %1452, label %1465

1452:                                             ; preds = %1449
  %1453 = mul i64 %1446, 32000
  %1454 = add i64 %1453, %1450
  %1455 = getelementptr float, ptr %1358, i64 %1454
  %1456 = load float, ptr %1455, align 4
  %1457 = getelementptr float, ptr %1424, i64 %1446
  %1458 = load float, ptr %1457, align 4
  %1459 = getelementptr i64, ptr %1437, i64 %1446
  %1460 = load i64, ptr %1459, align 4
  %1461 = fcmp ogt float %1456, %1458
  %1462 = select i1 %1461, float %1456, float %1458
  %1463 = select i1 %1461, i64 %1450, i64 %1460
  store float %1462, ptr %1457, align 4
  store i64 %1463, ptr %1459, align 4
  %1464 = add i64 %1450, 1
  br label %1449

1465:                                             ; preds = %1449
  %1466 = add i64 %1446, 1
  br label %1445

1467:                                             ; preds = %1445
  %1468 = load i64, ptr %1437, align 4
  call void @decode(i64 %36, i64 %1468)
  br label %31

1469:                                             ; preds = %31
  call void @end(i64 30)
  call void @free_tokenizer()
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.pow.f32(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.cos.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sin.f32(float) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
