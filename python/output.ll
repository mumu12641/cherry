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

31:                                               ; preds = %2191, %0
  %32 = phi i64 [ %2192, %2191 ], [ 1, %0 ]
  %33 = phi i64 [ %38, %2191 ], [ 0, %0 ]
  %34 = icmp slt i64 %33, 128
  br i1 %34, label %35, label %2193

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

56:                                               ; preds = %1922, %35
  %57 = phi i64 [ %1923, %1922 ], [ 0, %35 ]
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %1893, %1922 ], [ %53, %35 ]
  %59 = icmp slt i64 %57, 12
  br i1 %59, label %60, label %1924

60:                                               ; preds = %56
  %61 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %62 = ptrtoint ptr %61 to i64
  %63 = add i64 %62, 63
  %64 = urem i64 %63, 64
  %65 = sub i64 %63, %64
  %66 = inttoptr i64 %65 to ptr
  br label %67

67:                                               ; preds = %70, %60
  %68 = phi i64 [ %72, %70 ], [ 0, %60 ]
  %69 = icmp slt i64 %68, 1
  br i1 %69, label %70, label %73

70:                                               ; preds = %67
  %71 = getelementptr float, ptr %66, i64 %68
  store float 0.000000e+00, ptr %71, align 4
  %72 = add i64 %68, 1
  br label %67

73:                                               ; preds = %67
  br label %74

74:                                               ; preds = %106, %73
  %75 = phi i64 [ %107, %106 ], [ 0, %73 ]
  %76 = icmp slt i64 %75, 768
  br i1 %76, label %77, label %108

77:                                               ; preds = %74
  br label %78

78:                                               ; preds = %104, %77
  %79 = phi i64 [ %105, %104 ], [ 0, %77 ]
  %80 = icmp slt i64 %79, 128
  br i1 %80, label %81, label %106

81:                                               ; preds = %78
  %82 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %83 = add i64 %75, %79
  br label %84

84:                                               ; preds = %102, %81
  %85 = phi i64 [ %103, %102 ], [ 0, %81 ]
  %86 = icmp slt i64 %85, 1
  br i1 %86, label %87, label %104

87:                                               ; preds = %84
  br label %88

88:                                               ; preds = %91, %87
  %89 = phi i64 [ %101, %91 ], [ 0, %87 ]
  %90 = icmp slt i64 %89, 32
  br i1 %90, label %91, label %102

91:                                               ; preds = %88
  %92 = getelementptr float, ptr %82, i64 %83
  %93 = mul i64 %85, 768
  %94 = add i64 %93, %89
  %95 = getelementptr float, ptr %92, i64 %94
  %96 = load float, ptr %95, align 4
  %97 = getelementptr float, ptr %66, i64 %85
  %98 = load float, ptr %97, align 4
  %99 = fmul float %96, %96
  %100 = fadd float %98, %99
  store float %100, ptr %97, align 4
  %101 = add i64 %89, 1
  br label %88

102:                                              ; preds = %88
  %103 = add i64 %85, 1
  br label %84

104:                                              ; preds = %84
  %105 = add i64 %79, 32
  br label %78

106:                                              ; preds = %78
  %107 = add i64 %75, 128
  br label %74

108:                                              ; preds = %74
  %109 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %110 = ptrtoint ptr %109 to i64
  %111 = add i64 %110, 63
  %112 = urem i64 %111, 64
  %113 = sub i64 %111, %112
  %114 = inttoptr i64 %113 to ptr
  br label %115

115:                                              ; preds = %118, %108
  %116 = phi i64 [ %126, %118 ], [ 0, %108 ]
  %117 = icmp slt i64 %116, 1
  br i1 %117, label %118, label %127

118:                                              ; preds = %115
  %119 = getelementptr float, ptr %66, i64 %116
  %120 = load float, ptr %119, align 4
  %121 = fdiv float %120, 7.680000e+02
  %122 = fadd float %121, 0x3EE4F8B580000000
  %123 = call float @llvm.sqrt.f32(float %122)
  %124 = fdiv float 1.000000e+00, %123
  %125 = getelementptr float, ptr %114, i64 %116
  store float %124, ptr %125, align 4
  %126 = add i64 %116, 1
  br label %115

127:                                              ; preds = %115
  %128 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %129 = ptrtoint ptr %128 to i64
  %130 = add i64 %129, 63
  %131 = urem i64 %130, 64
  %132 = sub i64 %130, %131
  %133 = inttoptr i64 %132 to ptr
  br label %134

134:                                              ; preds = %167, %127
  %135 = phi i64 [ %168, %167 ], [ 0, %127 ]
  %136 = icmp slt i64 %135, 768
  br i1 %136, label %137, label %169

137:                                              ; preds = %134
  %138 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %139 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %140 = mul i64 %57, 768
  %141 = add i64 %140, %135
  br label %142

142:                                              ; preds = %165, %137
  %143 = phi i64 [ %166, %165 ], [ 0, %137 ]
  %144 = icmp slt i64 %143, 1
  br i1 %144, label %145, label %167

145:                                              ; preds = %142
  br label %146

146:                                              ; preds = %149, %145
  %147 = phi i64 [ %164, %149 ], [ 0, %145 ]
  %148 = icmp slt i64 %147, 32
  br i1 %148, label %149, label %165

149:                                              ; preds = %146
  %150 = getelementptr float, ptr %138, i64 %135
  %151 = mul i64 %143, 768
  %152 = add i64 %151, %147
  %153 = getelementptr float, ptr %150, i64 %152
  %154 = load float, ptr %153, align 4
  %155 = getelementptr float, ptr %114, i64 %143
  %156 = load float, ptr %155, align 4
  %157 = getelementptr float, ptr %139, i64 %141
  %158 = getelementptr float, ptr %157, i64 %147
  %159 = load float, ptr %158, align 4
  %160 = fmul float %154, %156
  %161 = fmul float %160, %159
  %162 = getelementptr float, ptr %133, i64 %135
  %163 = getelementptr float, ptr %162, i64 %152
  store float %161, ptr %163, align 4
  %164 = add i64 %147, 1
  br label %146

165:                                              ; preds = %146
  %166 = add i64 %143, 1
  br label %142

167:                                              ; preds = %142
  %168 = add i64 %135, 32
  br label %134

169:                                              ; preds = %134
  %170 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %171 = ptrtoint ptr %170 to i64
  %172 = add i64 %171, 63
  %173 = urem i64 %172, 64
  %174 = sub i64 %172, %173
  %175 = inttoptr i64 %174 to ptr
  br label %176

176:                                              ; preds = %195, %169
  %177 = phi i64 [ %196, %195 ], [ 0, %169 ]
  %178 = icmp slt i64 %177, 768
  br i1 %178, label %179, label %197

179:                                              ; preds = %176
  br label %180

180:                                              ; preds = %193, %179
  %181 = phi i64 [ %194, %193 ], [ 0, %179 ]
  %182 = icmp slt i64 %181, 1
  br i1 %182, label %183, label %195

183:                                              ; preds = %180
  br label %184

184:                                              ; preds = %187, %183
  %185 = phi i64 [ %192, %187 ], [ 0, %183 ]
  %186 = icmp slt i64 %185, 32
  br i1 %186, label %187, label %193

187:                                              ; preds = %184
  %188 = getelementptr float, ptr %175, i64 %177
  %189 = mul i64 %181, 768
  %190 = add i64 %189, %185
  %191 = getelementptr float, ptr %188, i64 %190
  store float 0.000000e+00, ptr %191, align 4
  %192 = add i64 %185, 1
  br label %184

193:                                              ; preds = %184
  %194 = add i64 %181, 1
  br label %180

195:                                              ; preds = %180
  %196 = add i64 %177, 32
  br label %176

197:                                              ; preds = %176
  %198 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %199 = ptrtoint ptr %198 to i64
  %200 = add i64 %199, 63
  %201 = urem i64 %200, 64
  %202 = sub i64 %200, %201
  %203 = inttoptr i64 %202 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %203, ptr %175, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %204

204:                                              ; preds = %269, %197
  %205 = phi i64 [ %270, %269 ], [ 0, %197 ]
  %206 = icmp slt i64 %205, 768
  br i1 %206, label %207, label %271

207:                                              ; preds = %204
  br label %208

208:                                              ; preds = %267, %207
  %209 = phi i64 [ %268, %267 ], [ 0, %207 ]
  %210 = icmp slt i64 %209, 768
  br i1 %210, label %211, label %269

211:                                              ; preds = %208
  br label %212

212:                                              ; preds = %265, %211
  %213 = phi i64 [ %266, %265 ], [ 0, %211 ]
  %214 = icmp slt i64 %213, 128
  br i1 %214, label %215, label %267

215:                                              ; preds = %212
  %216 = add i64 %205, %213
  br label %217

217:                                              ; preds = %263, %215
  %218 = phi i64 [ %264, %263 ], [ 0, %215 ]
  %219 = icmp slt i64 %218, 128
  br i1 %219, label %220, label %265

220:                                              ; preds = %217
  %221 = add i64 %209, %218
  %222 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %9, 1
  %223 = mul i64 %57, 589824
  %224 = mul i64 %209, 768
  %225 = add i64 %223, %224
  %226 = mul i64 %218, 768
  %227 = add i64 %225, %226
  %228 = add i64 %227, %205
  %229 = add i64 %228, %213
  br label %230

230:                                              ; preds = %261, %220
  %231 = phi i64 [ %262, %261 ], [ 0, %220 ]
  %232 = icmp slt i64 %231, 1
  br i1 %232, label %233, label %263

233:                                              ; preds = %230
  br label %234

234:                                              ; preds = %259, %233
  %235 = phi i64 [ %260, %259 ], [ 0, %233 ]
  %236 = icmp slt i64 %235, 32
  br i1 %236, label %237, label %261

237:                                              ; preds = %234
  br label %238

238:                                              ; preds = %241, %237
  %239 = phi i64 [ %258, %241 ], [ 0, %237 ]
  %240 = icmp slt i64 %239, 32
  br i1 %240, label %241, label %259

241:                                              ; preds = %238
  %242 = getelementptr float, ptr %133, i64 %221
  %243 = mul i64 %231, 768
  %244 = add i64 %243, %239
  %245 = getelementptr float, ptr %242, i64 %244
  %246 = load float, ptr %245, align 4
  %247 = getelementptr float, ptr %222, i64 %229
  %248 = mul i64 %239, 768
  %249 = add i64 %248, %235
  %250 = getelementptr float, ptr %247, i64 %249
  %251 = load float, ptr %250, align 4
  %252 = getelementptr float, ptr %203, i64 %216
  %253 = add i64 %243, %235
  %254 = getelementptr float, ptr %252, i64 %253
  %255 = load float, ptr %254, align 4
  %256 = fmul float %246, %251
  %257 = fadd float %255, %256
  store float %257, ptr %254, align 4
  %258 = add i64 %239, 1
  br label %238

259:                                              ; preds = %238
  %260 = add i64 %235, 1
  br label %234

261:                                              ; preds = %234
  %262 = add i64 %231, 1
  br label %230

263:                                              ; preds = %230
  %264 = add i64 %218, 32
  br label %217

265:                                              ; preds = %217
  %266 = add i64 %213, 32
  br label %212

267:                                              ; preds = %212
  %268 = add i64 %209, 128
  br label %208

269:                                              ; preds = %208
  %270 = add i64 %205, 128
  br label %204

271:                                              ; preds = %204
  %272 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %273 = ptrtoint ptr %272 to i64
  %274 = add i64 %273, 63
  %275 = urem i64 %274, 64
  %276 = sub i64 %274, %275
  %277 = inttoptr i64 %276 to ptr
  br label %278

278:                                              ; preds = %297, %271
  %279 = phi i64 [ %298, %297 ], [ 0, %271 ]
  %280 = icmp slt i64 %279, 768
  br i1 %280, label %281, label %299

281:                                              ; preds = %278
  br label %282

282:                                              ; preds = %295, %281
  %283 = phi i64 [ %296, %295 ], [ 0, %281 ]
  %284 = icmp slt i64 %283, 1
  br i1 %284, label %285, label %297

285:                                              ; preds = %282
  br label %286

286:                                              ; preds = %289, %285
  %287 = phi i64 [ %294, %289 ], [ 0, %285 ]
  %288 = icmp slt i64 %287, 32
  br i1 %288, label %289, label %295

289:                                              ; preds = %286
  %290 = getelementptr float, ptr %277, i64 %279
  %291 = mul i64 %283, 768
  %292 = add i64 %291, %287
  %293 = getelementptr float, ptr %290, i64 %292
  store float 0.000000e+00, ptr %293, align 4
  %294 = add i64 %287, 1
  br label %286

295:                                              ; preds = %286
  %296 = add i64 %283, 1
  br label %282

297:                                              ; preds = %282
  %298 = add i64 %279, 32
  br label %278

299:                                              ; preds = %278
  %300 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %301 = ptrtoint ptr %300 to i64
  %302 = add i64 %301, 63
  %303 = urem i64 %302, 64
  %304 = sub i64 %302, %303
  %305 = inttoptr i64 %304 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %305, ptr %277, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %306

306:                                              ; preds = %371, %299
  %307 = phi i64 [ %372, %371 ], [ 0, %299 ]
  %308 = icmp slt i64 %307, 768
  br i1 %308, label %309, label %373

309:                                              ; preds = %306
  br label %310

310:                                              ; preds = %369, %309
  %311 = phi i64 [ %370, %369 ], [ 0, %309 ]
  %312 = icmp slt i64 %311, 768
  br i1 %312, label %313, label %371

313:                                              ; preds = %310
  br label %314

314:                                              ; preds = %367, %313
  %315 = phi i64 [ %368, %367 ], [ 0, %313 ]
  %316 = icmp slt i64 %315, 128
  br i1 %316, label %317, label %369

317:                                              ; preds = %314
  %318 = add i64 %307, %315
  br label %319

319:                                              ; preds = %365, %317
  %320 = phi i64 [ %366, %365 ], [ 0, %317 ]
  %321 = icmp slt i64 %320, 128
  br i1 %321, label %322, label %367

322:                                              ; preds = %319
  %323 = add i64 %311, %320
  %324 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 1
  %325 = mul i64 %57, 589824
  %326 = mul i64 %311, 768
  %327 = add i64 %325, %326
  %328 = mul i64 %320, 768
  %329 = add i64 %327, %328
  %330 = add i64 %329, %307
  %331 = add i64 %330, %315
  br label %332

332:                                              ; preds = %363, %322
  %333 = phi i64 [ %364, %363 ], [ 0, %322 ]
  %334 = icmp slt i64 %333, 1
  br i1 %334, label %335, label %365

335:                                              ; preds = %332
  br label %336

336:                                              ; preds = %361, %335
  %337 = phi i64 [ %362, %361 ], [ 0, %335 ]
  %338 = icmp slt i64 %337, 32
  br i1 %338, label %339, label %363

339:                                              ; preds = %336
  br label %340

340:                                              ; preds = %343, %339
  %341 = phi i64 [ %360, %343 ], [ 0, %339 ]
  %342 = icmp slt i64 %341, 32
  br i1 %342, label %343, label %361

343:                                              ; preds = %340
  %344 = getelementptr float, ptr %133, i64 %323
  %345 = mul i64 %333, 768
  %346 = add i64 %345, %341
  %347 = getelementptr float, ptr %344, i64 %346
  %348 = load float, ptr %347, align 4
  %349 = getelementptr float, ptr %324, i64 %331
  %350 = mul i64 %341, 768
  %351 = add i64 %350, %337
  %352 = getelementptr float, ptr %349, i64 %351
  %353 = load float, ptr %352, align 4
  %354 = getelementptr float, ptr %305, i64 %318
  %355 = add i64 %345, %337
  %356 = getelementptr float, ptr %354, i64 %355
  %357 = load float, ptr %356, align 4
  %358 = fmul float %348, %353
  %359 = fadd float %357, %358
  store float %359, ptr %356, align 4
  %360 = add i64 %341, 1
  br label %340

361:                                              ; preds = %340
  %362 = add i64 %337, 1
  br label %336

363:                                              ; preds = %336
  %364 = add i64 %333, 1
  br label %332

365:                                              ; preds = %332
  %366 = add i64 %320, 32
  br label %319

367:                                              ; preds = %319
  %368 = add i64 %315, 32
  br label %314

369:                                              ; preds = %314
  %370 = add i64 %311, 128
  br label %310

371:                                              ; preds = %310
  %372 = add i64 %307, 128
  br label %306

373:                                              ; preds = %306
  %374 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %375 = ptrtoint ptr %374 to i64
  %376 = add i64 %375, 63
  %377 = urem i64 %376, 64
  %378 = sub i64 %376, %377
  %379 = inttoptr i64 %378 to ptr
  br label %380

380:                                              ; preds = %399, %373
  %381 = phi i64 [ %400, %399 ], [ 0, %373 ]
  %382 = icmp slt i64 %381, 768
  br i1 %382, label %383, label %401

383:                                              ; preds = %380
  br label %384

384:                                              ; preds = %397, %383
  %385 = phi i64 [ %398, %397 ], [ 0, %383 ]
  %386 = icmp slt i64 %385, 1
  br i1 %386, label %387, label %399

387:                                              ; preds = %384
  br label %388

388:                                              ; preds = %391, %387
  %389 = phi i64 [ %396, %391 ], [ 0, %387 ]
  %390 = icmp slt i64 %389, 32
  br i1 %390, label %391, label %397

391:                                              ; preds = %388
  %392 = getelementptr float, ptr %379, i64 %381
  %393 = mul i64 %385, 768
  %394 = add i64 %393, %389
  %395 = getelementptr float, ptr %392, i64 %394
  store float 0.000000e+00, ptr %395, align 4
  %396 = add i64 %389, 1
  br label %388

397:                                              ; preds = %388
  %398 = add i64 %385, 1
  br label %384

399:                                              ; preds = %384
  %400 = add i64 %381, 32
  br label %380

401:                                              ; preds = %380
  %402 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %403 = ptrtoint ptr %402 to i64
  %404 = add i64 %403, 63
  %405 = urem i64 %404, 64
  %406 = sub i64 %404, %405
  %407 = inttoptr i64 %406 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %407, ptr %379, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %408

408:                                              ; preds = %473, %401
  %409 = phi i64 [ %474, %473 ], [ 0, %401 ]
  %410 = icmp slt i64 %409, 768
  br i1 %410, label %411, label %475

411:                                              ; preds = %408
  br label %412

412:                                              ; preds = %471, %411
  %413 = phi i64 [ %472, %471 ], [ 0, %411 ]
  %414 = icmp slt i64 %413, 768
  br i1 %414, label %415, label %473

415:                                              ; preds = %412
  br label %416

416:                                              ; preds = %469, %415
  %417 = phi i64 [ %470, %469 ], [ 0, %415 ]
  %418 = icmp slt i64 %417, 128
  br i1 %418, label %419, label %471

419:                                              ; preds = %416
  %420 = add i64 %409, %417
  br label %421

421:                                              ; preds = %467, %419
  %422 = phi i64 [ %468, %467 ], [ 0, %419 ]
  %423 = icmp slt i64 %422, 128
  br i1 %423, label %424, label %469

424:                                              ; preds = %421
  %425 = add i64 %413, %422
  %426 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %11, 1
  %427 = mul i64 %57, 589824
  %428 = mul i64 %413, 768
  %429 = add i64 %427, %428
  %430 = mul i64 %422, 768
  %431 = add i64 %429, %430
  %432 = add i64 %431, %409
  %433 = add i64 %432, %417
  br label %434

434:                                              ; preds = %465, %424
  %435 = phi i64 [ %466, %465 ], [ 0, %424 ]
  %436 = icmp slt i64 %435, 1
  br i1 %436, label %437, label %467

437:                                              ; preds = %434
  br label %438

438:                                              ; preds = %463, %437
  %439 = phi i64 [ %464, %463 ], [ 0, %437 ]
  %440 = icmp slt i64 %439, 32
  br i1 %440, label %441, label %465

441:                                              ; preds = %438
  br label %442

442:                                              ; preds = %445, %441
  %443 = phi i64 [ %462, %445 ], [ 0, %441 ]
  %444 = icmp slt i64 %443, 32
  br i1 %444, label %445, label %463

445:                                              ; preds = %442
  %446 = getelementptr float, ptr %133, i64 %425
  %447 = mul i64 %435, 768
  %448 = add i64 %447, %443
  %449 = getelementptr float, ptr %446, i64 %448
  %450 = load float, ptr %449, align 4
  %451 = getelementptr float, ptr %426, i64 %433
  %452 = mul i64 %443, 768
  %453 = add i64 %452, %439
  %454 = getelementptr float, ptr %451, i64 %453
  %455 = load float, ptr %454, align 4
  %456 = getelementptr float, ptr %407, i64 %420
  %457 = add i64 %447, %439
  %458 = getelementptr float, ptr %456, i64 %457
  %459 = load float, ptr %458, align 4
  %460 = fmul float %450, %455
  %461 = fadd float %459, %460
  store float %461, ptr %458, align 4
  %462 = add i64 %443, 1
  br label %442

463:                                              ; preds = %442
  %464 = add i64 %439, 1
  br label %438

465:                                              ; preds = %438
  %466 = add i64 %435, 1
  br label %434

467:                                              ; preds = %434
  %468 = add i64 %422, 32
  br label %421

469:                                              ; preds = %421
  %470 = add i64 %417, 32
  br label %416

471:                                              ; preds = %416
  %472 = add i64 %413, 128
  br label %412

473:                                              ; preds = %412
  %474 = add i64 %409, 128
  br label %408

475:                                              ; preds = %408
  %476 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %477 = ptrtoint ptr %476 to i64
  %478 = add i64 %477, 63
  %479 = urem i64 %478, 64
  %480 = sub i64 %478, %479
  %481 = inttoptr i64 %480 to ptr
  %482 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %483 = ptrtoint ptr %482 to i64
  %484 = add i64 %483, 63
  %485 = urem i64 %484, 64
  %486 = sub i64 %484, %485
  %487 = inttoptr i64 %486 to ptr
  br label %488

488:                                              ; preds = %491, %475
  %489 = phi i64 [ %501, %491 ], [ 0, %475 ]
  %490 = icmp slt i64 %489, 32
  br i1 %490, label %491, label %502

491:                                              ; preds = %488
  %492 = uitofp i64 %489 to float
  %493 = fmul float %492, -2.000000e+00
  %494 = fdiv float %493, 6.400000e+01
  %495 = call float @llvm.pow.f32(float 1.000000e+04, float %494)
  %496 = fmul float %55, %495
  %497 = call float @llvm.cos.f32(float %496)
  %498 = call float @llvm.sin.f32(float %496)
  %499 = getelementptr float, ptr %481, i64 %489
  store float %497, ptr %499, align 4
  %500 = getelementptr float, ptr %487, i64 %489
  store float %498, ptr %500, align 4
  %501 = add i64 %489, 1
  br label %488

502:                                              ; preds = %488
  %503 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %504 = ptrtoint ptr %503 to i64
  %505 = add i64 %504, 63
  %506 = urem i64 %505, 64
  %507 = sub i64 %505, %506
  %508 = inttoptr i64 %507 to ptr
  %509 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %510 = ptrtoint ptr %509 to i64
  %511 = add i64 %510, 63
  %512 = urem i64 %511, 64
  %513 = sub i64 %511, %512
  %514 = inttoptr i64 %513 to ptr
  br label %515

515:                                              ; preds = %556, %502
  %516 = phi i64 [ %557, %556 ], [ 0, %502 ]
  %517 = icmp slt i64 %516, 1
  br i1 %517, label %518, label %558

518:                                              ; preds = %515
  br label %519

519:                                              ; preds = %554, %518
  %520 = phi i64 [ %555, %554 ], [ 0, %518 ]
  %521 = icmp slt i64 %520, 12
  br i1 %521, label %522, label %556

522:                                              ; preds = %519
  br label %523

523:                                              ; preds = %526, %522
  %524 = phi i64 [ %553, %526 ], [ 0, %522 ]
  %525 = icmp slt i64 %524, 32
  br i1 %525, label %526, label %554

526:                                              ; preds = %523
  %527 = mul i64 %516, 768
  %528 = mul i64 %520, 64
  %529 = add i64 %527, %528
  %530 = mul i64 %524, 2
  %531 = add i64 %529, %530
  %532 = getelementptr float, ptr %203, i64 %531
  %533 = load float, ptr %532, align 4
  %534 = getelementptr float, ptr %203, i32 1
  %535 = getelementptr float, ptr %534, i64 %531
  %536 = load float, ptr %535, align 4
  %537 = getelementptr float, ptr %481, i64 %524
  %538 = load float, ptr %537, align 4
  %539 = getelementptr float, ptr %487, i64 %524
  %540 = load float, ptr %539, align 4
  %541 = fmul float %533, %538
  %542 = fmul float %536, %540
  %543 = fsub float %541, %542
  %544 = fmul float %536, %538
  %545 = fmul float %533, %540
  %546 = fadd float %544, %545
  %547 = mul i64 %516, 384
  %548 = mul i64 %520, 32
  %549 = add i64 %547, %548
  %550 = add i64 %549, %524
  %551 = getelementptr float, ptr %508, i64 %550
  store float %543, ptr %551, align 4
  %552 = getelementptr float, ptr %514, i64 %550
  store float %546, ptr %552, align 4
  %553 = add i64 %524, 1
  br label %523

554:                                              ; preds = %523
  %555 = add i64 %520, 1
  br label %519

556:                                              ; preds = %519
  %557 = add i64 %516, 1
  br label %515

558:                                              ; preds = %515
  %559 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %503, 0
  %560 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %559, ptr %508, 1
  %561 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %560, i64 0, 2
  %562 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %561, i64 1, 3, 0
  %563 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %562, i64 384, 4, 0
  %564 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %563, i64 12, 3, 1
  %565 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %564, i64 32, 4, 1
  %566 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %565, i64 32, 3, 2
  %567 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %566, i64 1, 4, 2
  %568 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %567, i64 1, 3, 3
  %569 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %568, i64 1, 4, 3
  %570 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %509, 0
  %571 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %570, ptr %514, 1
  %572 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %571, i64 0, 2
  %573 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %572, i64 1, 3, 0
  %574 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %573, i64 384, 4, 0
  %575 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %574, i64 12, 3, 1
  %576 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %575, i64 32, 4, 1
  %577 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %576, i64 32, 3, 2
  %578 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %577, i64 1, 4, 2
  %579 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %578, i64 1, 3, 3
  %580 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %579, i64 1, 4, 3
  %581 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %582 = ptrtoint ptr %581 to i64
  %583 = add i64 %582, 63
  %584 = urem i64 %583, 64
  %585 = sub i64 %583, %584
  %586 = inttoptr i64 %585 to ptr
  %587 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %581, 0
  %588 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %587, ptr %586, 1
  %589 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %588, i64 0, 2
  %590 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %589, i64 1, 3, 0
  %591 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %590, i64 768, 4, 0
  %592 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %591, i64 12, 3, 1
  %593 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %592, i64 64, 4, 1
  %594 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %593, i64 32, 3, 2
  %595 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %594, i64 2, 4, 2
  %596 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %595, i64 1, 3, 3
  %597 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %596, i64 1, 4, 3
  %598 = call ptr @llvm.stacksave.p0()
  %599 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %569, ptr %599, align 8
  %600 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %599, 1
  %601 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %597, ptr %601, align 8
  %602 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %601, 1
  %603 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %600, ptr %603, align 8
  %604 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %602, ptr %604, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %603, ptr %604)
  call void @llvm.stackrestore.p0(ptr %598)
  %605 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %606 = ptrtoint ptr %605 to i64
  %607 = add i64 %606, 63
  %608 = urem i64 %607, 64
  %609 = sub i64 %607, %608
  %610 = inttoptr i64 %609 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %610, ptr %586, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %611 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %605, 0
  %612 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %611, ptr %610, 1
  %613 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %612, i64 1, 2
  %614 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %613, i64 1, 3, 0
  %615 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %614, i64 768, 4, 0
  %616 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %615, i64 12, 3, 1
  %617 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %616, i64 64, 4, 1
  %618 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %617, i64 32, 3, 2
  %619 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %618, i64 2, 4, 2
  %620 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %619, i64 1, 3, 3
  %621 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %620, i64 1, 4, 3
  %622 = call ptr @llvm.stacksave.p0()
  %623 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %580, ptr %623, align 8
  %624 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %623, 1
  %625 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %621, ptr %625, align 8
  %626 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %625, 1
  %627 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %624, ptr %627, align 8
  %628 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %626, ptr %628, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %627, ptr %628)
  call void @llvm.stackrestore.p0(ptr %622)
  %629 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %630 = ptrtoint ptr %629 to i64
  %631 = add i64 %630, 63
  %632 = urem i64 %631, 64
  %633 = sub i64 %631, %632
  %634 = inttoptr i64 %633 to ptr
  %635 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %636 = ptrtoint ptr %635 to i64
  %637 = add i64 %636, 63
  %638 = urem i64 %637, 64
  %639 = sub i64 %637, %638
  %640 = inttoptr i64 %639 to ptr
  br label %641

641:                                              ; preds = %644, %558
  %642 = phi i64 [ %654, %644 ], [ 0, %558 ]
  %643 = icmp slt i64 %642, 32
  br i1 %643, label %644, label %655

644:                                              ; preds = %641
  %645 = uitofp i64 %642 to float
  %646 = fmul float %645, -2.000000e+00
  %647 = fdiv float %646, 6.400000e+01
  %648 = call float @llvm.pow.f32(float 1.000000e+04, float %647)
  %649 = fmul float %55, %648
  %650 = call float @llvm.cos.f32(float %649)
  %651 = call float @llvm.sin.f32(float %649)
  %652 = getelementptr float, ptr %634, i64 %642
  store float %650, ptr %652, align 4
  %653 = getelementptr float, ptr %640, i64 %642
  store float %651, ptr %653, align 4
  %654 = add i64 %642, 1
  br label %641

655:                                              ; preds = %641
  %656 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %657 = ptrtoint ptr %656 to i64
  %658 = add i64 %657, 63
  %659 = urem i64 %658, 64
  %660 = sub i64 %658, %659
  %661 = inttoptr i64 %660 to ptr
  %662 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %663 = ptrtoint ptr %662 to i64
  %664 = add i64 %663, 63
  %665 = urem i64 %664, 64
  %666 = sub i64 %664, %665
  %667 = inttoptr i64 %666 to ptr
  br label %668

668:                                              ; preds = %709, %655
  %669 = phi i64 [ %710, %709 ], [ 0, %655 ]
  %670 = icmp slt i64 %669, 1
  br i1 %670, label %671, label %711

671:                                              ; preds = %668
  br label %672

672:                                              ; preds = %707, %671
  %673 = phi i64 [ %708, %707 ], [ 0, %671 ]
  %674 = icmp slt i64 %673, 12
  br i1 %674, label %675, label %709

675:                                              ; preds = %672
  br label %676

676:                                              ; preds = %679, %675
  %677 = phi i64 [ %706, %679 ], [ 0, %675 ]
  %678 = icmp slt i64 %677, 32
  br i1 %678, label %679, label %707

679:                                              ; preds = %676
  %680 = mul i64 %669, 768
  %681 = mul i64 %673, 64
  %682 = add i64 %680, %681
  %683 = mul i64 %677, 2
  %684 = add i64 %682, %683
  %685 = getelementptr float, ptr %305, i64 %684
  %686 = load float, ptr %685, align 4
  %687 = getelementptr float, ptr %305, i32 1
  %688 = getelementptr float, ptr %687, i64 %684
  %689 = load float, ptr %688, align 4
  %690 = getelementptr float, ptr %634, i64 %677
  %691 = load float, ptr %690, align 4
  %692 = getelementptr float, ptr %640, i64 %677
  %693 = load float, ptr %692, align 4
  %694 = fmul float %686, %691
  %695 = fmul float %689, %693
  %696 = fsub float %694, %695
  %697 = fmul float %689, %691
  %698 = fmul float %686, %693
  %699 = fadd float %697, %698
  %700 = mul i64 %669, 384
  %701 = mul i64 %673, 32
  %702 = add i64 %700, %701
  %703 = add i64 %702, %677
  %704 = getelementptr float, ptr %661, i64 %703
  store float %696, ptr %704, align 4
  %705 = getelementptr float, ptr %667, i64 %703
  store float %699, ptr %705, align 4
  %706 = add i64 %677, 1
  br label %676

707:                                              ; preds = %676
  %708 = add i64 %673, 1
  br label %672

709:                                              ; preds = %672
  %710 = add i64 %669, 1
  br label %668

711:                                              ; preds = %668
  %712 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %656, 0
  %713 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %712, ptr %661, 1
  %714 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %713, i64 0, 2
  %715 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %714, i64 1, 3, 0
  %716 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %715, i64 384, 4, 0
  %717 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %716, i64 12, 3, 1
  %718 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %717, i64 32, 4, 1
  %719 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %718, i64 32, 3, 2
  %720 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %719, i64 1, 4, 2
  %721 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %720, i64 1, 3, 3
  %722 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %721, i64 1, 4, 3
  %723 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %662, 0
  %724 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %723, ptr %667, 1
  %725 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %724, i64 0, 2
  %726 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %725, i64 1, 3, 0
  %727 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %726, i64 384, 4, 0
  %728 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %727, i64 12, 3, 1
  %729 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %728, i64 32, 4, 1
  %730 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %729, i64 32, 3, 2
  %731 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %730, i64 1, 4, 2
  %732 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %731, i64 1, 3, 3
  %733 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %732, i64 1, 4, 3
  %734 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %735 = ptrtoint ptr %734 to i64
  %736 = add i64 %735, 63
  %737 = urem i64 %736, 64
  %738 = sub i64 %736, %737
  %739 = inttoptr i64 %738 to ptr
  %740 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %734, 0
  %741 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %740, ptr %739, 1
  %742 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %741, i64 0, 2
  %743 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %742, i64 1, 3, 0
  %744 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %743, i64 768, 4, 0
  %745 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %744, i64 12, 3, 1
  %746 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %745, i64 64, 4, 1
  %747 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %746, i64 32, 3, 2
  %748 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %747, i64 2, 4, 2
  %749 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %748, i64 1, 3, 3
  %750 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %749, i64 1, 4, 3
  %751 = call ptr @llvm.stacksave.p0()
  %752 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %722, ptr %752, align 8
  %753 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %752, 1
  %754 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %750, ptr %754, align 8
  %755 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %754, 1
  %756 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %753, ptr %756, align 8
  %757 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %755, ptr %757, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %756, ptr %757)
  call void @llvm.stackrestore.p0(ptr %751)
  %758 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %759 = ptrtoint ptr %758 to i64
  %760 = add i64 %759, 63
  %761 = urem i64 %760, 64
  %762 = sub i64 %760, %761
  %763 = inttoptr i64 %762 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %763, ptr %739, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %764 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %758, 0
  %765 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %764, ptr %763, 1
  %766 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %765, i64 1, 2
  %767 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %766, i64 1, 3, 0
  %768 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %767, i64 768, 4, 0
  %769 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %768, i64 12, 3, 1
  %770 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %769, i64 64, 4, 1
  %771 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %770, i64 32, 3, 2
  %772 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %771, i64 2, 4, 2
  %773 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %772, i64 1, 3, 3
  %774 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %773, i64 1, 4, 3
  %775 = call ptr @llvm.stacksave.p0()
  %776 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %733, ptr %776, align 8
  %777 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %776, 1
  %778 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %774, ptr %778, align 8
  %779 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %778, 1
  %780 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %777, ptr %780, align 8
  %781 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %779, ptr %781, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %780, ptr %781)
  call void @llvm.stackrestore.p0(ptr %775)
  %782 = mul i64 %57, 786432
  %783 = mul i64 %37, 768
  %784 = add i64 %782, %783
  %785 = getelementptr float, ptr %24, i64 %784
  call void @llvm.memcpy.p0.p0.i64(ptr %785, ptr %763, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %786 = getelementptr float, ptr %30, i64 %784
  call void @llvm.memcpy.p0.p0.i64(ptr %786, ptr %407, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %787 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %788 = ptrtoint ptr %787 to i64
  %789 = add i64 %788, 63
  %790 = urem i64 %789, 64
  %791 = sub i64 %789, %790
  %792 = inttoptr i64 %791 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %792, ptr @__constant_1x12x64xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %793

793:                                              ; preds = %1276, %711
  %794 = phi i64 [ %1279, %1276 ], [ 0, %711 ]
  %795 = icmp slt i64 %794, 12
  br i1 %795, label %796, label %1280

796:                                              ; preds = %793
  %797 = mul i64 %794, 64
  %798 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 65536) to i64), i64 64))
  %799 = ptrtoint ptr %798 to i64
  %800 = add i64 %799, 63
  %801 = urem i64 %800, 64
  %802 = sub i64 %800, %801
  %803 = inttoptr i64 %802 to ptr
  br label %804

804:                                              ; preds = %840, %796
  %805 = phi i64 [ %841, %840 ], [ 0, %796 ]
  %806 = icmp slt i64 %805, 64
  br i1 %806, label %807, label %842

807:                                              ; preds = %804
  br label %808

808:                                              ; preds = %838, %807
  %809 = phi i64 [ %839, %838 ], [ 0, %807 ]
  %810 = icmp slt i64 %809, 1024
  br i1 %810, label %811, label %840

811:                                              ; preds = %808
  %812 = mul i64 %809, 768
  %813 = add i64 %782, %812
  %814 = add i64 %813, %797
  %815 = add i64 %814, %805
  %816 = mul i64 %805, 1024
  %817 = add i64 %816, %809
  br label %818

818:                                              ; preds = %836, %811
  %819 = phi i64 [ %837, %836 ], [ 0, %811 ]
  %820 = icmp slt i64 %819, 32
  br i1 %820, label %821, label %838

821:                                              ; preds = %818
  br label %822

822:                                              ; preds = %825, %821
  %823 = phi i64 [ %835, %825 ], [ 0, %821 ]
  %824 = icmp slt i64 %823, 32
  br i1 %824, label %825, label %836

825:                                              ; preds = %822
  %826 = getelementptr float, ptr %24, i64 %815
  %827 = mul i64 %823, 768
  %828 = add i64 %827, %819
  %829 = getelementptr float, ptr %826, i64 %828
  %830 = load float, ptr %829, align 4
  %831 = getelementptr float, ptr %803, i64 %817
  %832 = mul i64 %819, 1024
  %833 = add i64 %832, %823
  %834 = getelementptr float, ptr %831, i64 %833
  store float %830, ptr %834, align 4
  %835 = add i64 %823, 1
  br label %822

836:                                              ; preds = %822
  %837 = add i64 %819, 1
  br label %818

838:                                              ; preds = %818
  %839 = add i64 %809, 32
  br label %808

840:                                              ; preds = %808
  %841 = add i64 %805, 32
  br label %804

842:                                              ; preds = %804
  %843 = mul i64 %38, 1
  %844 = getelementptr float, ptr null, i64 %843
  %845 = ptrtoint ptr %844 to i64
  %846 = add i64 %845, 64
  %847 = call ptr @malloc(i64 %846)
  %848 = ptrtoint ptr %847 to i64
  %849 = add i64 %848, 63
  %850 = urem i64 %849, 64
  %851 = sub i64 %849, %850
  %852 = inttoptr i64 %851 to ptr
  %853 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %847, 0
  %854 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %853, ptr %852, 1
  %855 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %854, i64 0, 2
  %856 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %855, i64 1, 3, 0
  %857 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %856, i64 %38, 3, 1
  %858 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %857, i64 %38, 4, 0
  %859 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %858, i64 1, 4, 1
  br label %860

860:                                              ; preds = %882, %842
  %861 = phi i64 [ %883, %882 ], [ 0, %842 ]
  %862 = icmp slt i64 %861, %38
  br i1 %862, label %863, label %884

863:                                              ; preds = %860
  %864 = mul i64 %861, -1
  %865 = add i64 %38, %864
  %866 = call i64 @llvm.smin.i64(i64 %865, i64 32)
  br label %867

867:                                              ; preds = %880, %863
  %868 = phi i64 [ %881, %880 ], [ 0, %863 ]
  %869 = icmp slt i64 %868, 1
  br i1 %869, label %870, label %882

870:                                              ; preds = %867
  br label %871

871:                                              ; preds = %874, %870
  %872 = phi i64 [ %879, %874 ], [ 0, %870 ]
  %873 = icmp slt i64 %872, %866
  br i1 %873, label %874, label %880

874:                                              ; preds = %871
  %875 = getelementptr float, ptr %852, i64 %861
  %876 = mul i64 %868, %38
  %877 = add i64 %876, %872
  %878 = getelementptr float, ptr %875, i64 %877
  store float 0.000000e+00, ptr %878, align 4
  %879 = add i64 %872, 1
  br label %871

880:                                              ; preds = %871
  %881 = add i64 %868, 1
  br label %867

882:                                              ; preds = %867
  %883 = add i64 %861, 32
  br label %860

884:                                              ; preds = %860
  br label %885

885:                                              ; preds = %949, %884
  %886 = phi i64 [ %950, %949 ], [ 0, %884 ]
  %887 = icmp slt i64 %886, %38
  br i1 %887, label %888, label %951

888:                                              ; preds = %885
  %889 = mul i64 %886, -1
  %890 = add i64 %38, %889
  %891 = call i64 @llvm.smin.i64(i64 %890, i64 128)
  br label %892

892:                                              ; preds = %947, %888
  %893 = phi i64 [ %948, %947 ], [ 0, %888 ]
  %894 = icmp slt i64 %893, %891
  br i1 %894, label %895, label %949

895:                                              ; preds = %892
  %896 = mul i64 %893, -1
  %897 = add i64 %891, %896
  %898 = call i64 @llvm.smin.i64(i64 %897, i64 32)
  %899 = add i64 %886, %893
  br label %900

900:                                              ; preds = %945, %895
  %901 = phi i64 [ %946, %945 ], [ 0, %895 ]
  %902 = icmp slt i64 %901, 64
  br i1 %902, label %903, label %947

903:                                              ; preds = %900
  %904 = mul i64 %901, -1
  %905 = add i64 %904, 64
  %906 = call i64 @llvm.smin.i64(i64 %905, i64 32)
  %907 = add i64 %797, %901
  %908 = mul i64 %901, 1024
  %909 = add i64 %908, %886
  %910 = add i64 %909, %893
  br label %911

911:                                              ; preds = %943, %903
  %912 = phi i64 [ %944, %943 ], [ 0, %903 ]
  %913 = icmp slt i64 %912, 1
  br i1 %913, label %914, label %945

914:                                              ; preds = %911
  br label %915

915:                                              ; preds = %941, %914
  %916 = phi i64 [ %942, %941 ], [ 0, %914 ]
  %917 = icmp slt i64 %916, %898
  br i1 %917, label %918, label %943

918:                                              ; preds = %915
  br label %919

919:                                              ; preds = %922, %918
  %920 = phi i64 [ %940, %922 ], [ 0, %918 ]
  %921 = icmp slt i64 %920, %906
  br i1 %921, label %922, label %941

922:                                              ; preds = %919
  %923 = getelementptr float, ptr %610, i64 %907
  %924 = mul i64 %912, 768
  %925 = add i64 %924, %920
  %926 = getelementptr float, ptr %923, i64 %925
  %927 = load float, ptr %926, align 4
  %928 = getelementptr float, ptr %803, i64 %910
  %929 = mul i64 %920, 1024
  %930 = add i64 %929, %916
  %931 = getelementptr float, ptr %928, i64 %930
  %932 = load float, ptr %931, align 4
  %933 = getelementptr float, ptr %852, i64 %899
  %934 = mul i64 %912, %38
  %935 = add i64 %934, %916
  %936 = getelementptr float, ptr %933, i64 %935
  %937 = load float, ptr %936, align 4
  %938 = fmul float %927, %932
  %939 = fadd float %937, %938
  store float %939, ptr %936, align 4
  %940 = add i64 %920, 1
  br label %919

941:                                              ; preds = %919
  %942 = add i64 %916, 1
  br label %915

943:                                              ; preds = %915
  %944 = add i64 %912, 1
  br label %911

945:                                              ; preds = %911
  %946 = add i64 %901, 32
  br label %900

947:                                              ; preds = %900
  %948 = add i64 %893, 32
  br label %892

949:                                              ; preds = %892
  %950 = add i64 %886, 128
  br label %885

951:                                              ; preds = %885
  %952 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %953 = ptrtoint ptr %952 to i64
  %954 = add i64 %953, 63
  %955 = urem i64 %954, 64
  %956 = sub i64 %954, %955
  %957 = inttoptr i64 %956 to ptr
  br label %958

958:                                              ; preds = %977, %951
  %959 = phi i64 [ %978, %977 ], [ 0, %951 ]
  %960 = icmp slt i64 %959, 1024
  br i1 %960, label %961, label %979

961:                                              ; preds = %958
  br label %962

962:                                              ; preds = %975, %961
  %963 = phi i64 [ %976, %975 ], [ 0, %961 ]
  %964 = icmp slt i64 %963, 1
  br i1 %964, label %965, label %977

965:                                              ; preds = %962
  br label %966

966:                                              ; preds = %969, %965
  %967 = phi i64 [ %974, %969 ], [ 0, %965 ]
  %968 = icmp slt i64 %967, 32
  br i1 %968, label %969, label %975

969:                                              ; preds = %966
  %970 = getelementptr float, ptr %957, i64 %959
  %971 = mul i64 %963, 1024
  %972 = add i64 %971, %967
  %973 = getelementptr float, ptr %970, i64 %972
  store float -1.000000e+09, ptr %973, align 4
  %974 = add i64 %967, 1
  br label %966

975:                                              ; preds = %966
  %976 = add i64 %963, 1
  br label %962

977:                                              ; preds = %962
  %978 = add i64 %959, 32
  br label %958

979:                                              ; preds = %958
  %980 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %952, 0
  %981 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %980, ptr %957, 1
  %982 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %981, i64 0, 2
  %983 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %982, i64 1, 3, 0
  %984 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %983, i64 1024, 4, 0
  %985 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %984, i64 %38, 3, 1
  %986 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %985, i64 1, 4, 1
  %987 = call ptr @llvm.stacksave.p0()
  %988 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %859, ptr %988, align 8
  %989 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %988, 1
  %990 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %986, ptr %990, align 8
  %991 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %990, 1
  %992 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %989, ptr %992, align 8
  %993 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %991, ptr %993, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %992, ptr %993)
  call void @llvm.stackrestore.p0(ptr %987)
  %994 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %995 = ptrtoint ptr %994 to i64
  %996 = add i64 %995, 63
  %997 = urem i64 %996, 64
  %998 = sub i64 %996, %997
  %999 = inttoptr i64 %998 to ptr
  br label %1000

1000:                                             ; preds = %1023, %979
  %1001 = phi i64 [ %1024, %1023 ], [ 0, %979 ]
  %1002 = icmp slt i64 %1001, 1024
  br i1 %1002, label %1003, label %1025

1003:                                             ; preds = %1000
  br label %1004

1004:                                             ; preds = %1021, %1003
  %1005 = phi i64 [ %1022, %1021 ], [ 0, %1003 ]
  %1006 = icmp slt i64 %1005, 1
  br i1 %1006, label %1007, label %1023

1007:                                             ; preds = %1004
  br label %1008

1008:                                             ; preds = %1011, %1007
  %1009 = phi i64 [ %1020, %1011 ], [ 0, %1007 ]
  %1010 = icmp slt i64 %1009, 32
  br i1 %1010, label %1011, label %1021

1011:                                             ; preds = %1008
  %1012 = getelementptr float, ptr %957, i64 %1001
  %1013 = mul i64 %1005, 1024
  %1014 = add i64 %1013, %1009
  %1015 = getelementptr float, ptr %1012, i64 %1014
  %1016 = load float, ptr %1015, align 4
  %1017 = fmul float %1016, 1.250000e-01
  %1018 = getelementptr float, ptr %999, i64 %1001
  %1019 = getelementptr float, ptr %1018, i64 %1014
  store float %1017, ptr %1019, align 4
  %1020 = add i64 %1009, 1
  br label %1008

1021:                                             ; preds = %1008
  %1022 = add i64 %1005, 1
  br label %1004

1023:                                             ; preds = %1004
  %1024 = add i64 %1001, 32
  br label %1000

1025:                                             ; preds = %1000
  %1026 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1027 = ptrtoint ptr %1026 to i64
  %1028 = add i64 %1027, 63
  %1029 = urem i64 %1028, 64
  %1030 = sub i64 %1028, %1029
  %1031 = inttoptr i64 %1030 to ptr
  br label %1032

1032:                                             ; preds = %1035, %1025
  %1033 = phi i64 [ %1037, %1035 ], [ 0, %1025 ]
  %1034 = icmp slt i64 %1033, 1
  br i1 %1034, label %1035, label %1038

1035:                                             ; preds = %1032
  %1036 = getelementptr float, ptr %1031, i64 %1033
  store float 0xFFF0000000000000, ptr %1036, align 4
  %1037 = add i64 %1033, 1
  br label %1032

1038:                                             ; preds = %1032
  br label %1039

1039:                                             ; preds = %1069, %1038
  %1040 = phi i64 [ %1070, %1069 ], [ 0, %1038 ]
  %1041 = icmp slt i64 %1040, 1024
  br i1 %1041, label %1042, label %1071

1042:                                             ; preds = %1039
  br label %1043

1043:                                             ; preds = %1067, %1042
  %1044 = phi i64 [ %1068, %1067 ], [ 0, %1042 ]
  %1045 = icmp slt i64 %1044, 128
  br i1 %1045, label %1046, label %1069

1046:                                             ; preds = %1043
  %1047 = add i64 %1040, %1044
  br label %1048

1048:                                             ; preds = %1065, %1046
  %1049 = phi i64 [ %1066, %1065 ], [ 0, %1046 ]
  %1050 = icmp slt i64 %1049, 1
  br i1 %1050, label %1051, label %1067

1051:                                             ; preds = %1048
  br label %1052

1052:                                             ; preds = %1055, %1051
  %1053 = phi i64 [ %1064, %1055 ], [ 0, %1051 ]
  %1054 = icmp slt i64 %1053, 32
  br i1 %1054, label %1055, label %1065

1055:                                             ; preds = %1052
  %1056 = getelementptr float, ptr %999, i64 %1047
  %1057 = mul i64 %1049, 1024
  %1058 = add i64 %1057, %1053
  %1059 = getelementptr float, ptr %1056, i64 %1058
  %1060 = load float, ptr %1059, align 4
  %1061 = getelementptr float, ptr %1031, i64 %1049
  %1062 = load float, ptr %1061, align 4
  %1063 = call float @llvm.maxnum.f32(float %1060, float %1062)
  store float %1063, ptr %1061, align 4
  %1064 = add i64 %1053, 1
  br label %1052

1065:                                             ; preds = %1052
  %1066 = add i64 %1049, 1
  br label %1048

1067:                                             ; preds = %1048
  %1068 = add i64 %1044, 32
  br label %1043

1069:                                             ; preds = %1043
  %1070 = add i64 %1040, 128
  br label %1039

1071:                                             ; preds = %1039
  %1072 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %1073 = ptrtoint ptr %1072 to i64
  %1074 = add i64 %1073, 63
  %1075 = urem i64 %1074, 64
  %1076 = sub i64 %1074, %1075
  %1077 = inttoptr i64 %1076 to ptr
  br label %1078

1078:                                             ; preds = %1104, %1071
  %1079 = phi i64 [ %1105, %1104 ], [ 0, %1071 ]
  %1080 = icmp slt i64 %1079, 1024
  br i1 %1080, label %1081, label %1106

1081:                                             ; preds = %1078
  br label %1082

1082:                                             ; preds = %1102, %1081
  %1083 = phi i64 [ %1103, %1102 ], [ 0, %1081 ]
  %1084 = icmp slt i64 %1083, 1
  br i1 %1084, label %1085, label %1104

1085:                                             ; preds = %1082
  br label %1086

1086:                                             ; preds = %1089, %1085
  %1087 = phi i64 [ %1101, %1089 ], [ 0, %1085 ]
  %1088 = icmp slt i64 %1087, 32
  br i1 %1088, label %1089, label %1102

1089:                                             ; preds = %1086
  %1090 = getelementptr float, ptr %999, i64 %1079
  %1091 = mul i64 %1083, 1024
  %1092 = add i64 %1091, %1087
  %1093 = getelementptr float, ptr %1090, i64 %1092
  %1094 = load float, ptr %1093, align 4
  %1095 = getelementptr float, ptr %1031, i64 %1083
  %1096 = load float, ptr %1095, align 4
  %1097 = fsub float %1094, %1096
  %1098 = call float @llvm.exp.f32(float %1097)
  %1099 = getelementptr float, ptr %1077, i64 %1079
  %1100 = getelementptr float, ptr %1099, i64 %1092
  store float %1098, ptr %1100, align 4
  %1101 = add i64 %1087, 1
  br label %1086

1102:                                             ; preds = %1086
  %1103 = add i64 %1083, 1
  br label %1082

1104:                                             ; preds = %1082
  %1105 = add i64 %1079, 32
  br label %1078

1106:                                             ; preds = %1078
  %1107 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1108 = ptrtoint ptr %1107 to i64
  %1109 = add i64 %1108, 63
  %1110 = urem i64 %1109, 64
  %1111 = sub i64 %1109, %1110
  %1112 = inttoptr i64 %1111 to ptr
  br label %1113

1113:                                             ; preds = %1116, %1106
  %1114 = phi i64 [ %1118, %1116 ], [ 0, %1106 ]
  %1115 = icmp slt i64 %1114, 1
  br i1 %1115, label %1116, label %1119

1116:                                             ; preds = %1113
  %1117 = getelementptr float, ptr %1112, i64 %1114
  store float 0.000000e+00, ptr %1117, align 4
  %1118 = add i64 %1114, 1
  br label %1113

1119:                                             ; preds = %1113
  br label %1120

1120:                                             ; preds = %1143, %1119
  %1121 = phi i64 [ %1144, %1143 ], [ 0, %1119 ]
  %1122 = icmp slt i64 %1121, 1024
  br i1 %1122, label %1123, label %1145

1123:                                             ; preds = %1120
  br label %1124

1124:                                             ; preds = %1141, %1123
  %1125 = phi i64 [ %1142, %1141 ], [ 0, %1123 ]
  %1126 = icmp slt i64 %1125, 1
  br i1 %1126, label %1127, label %1143

1127:                                             ; preds = %1124
  br label %1128

1128:                                             ; preds = %1131, %1127
  %1129 = phi i64 [ %1140, %1131 ], [ 0, %1127 ]
  %1130 = icmp slt i64 %1129, 32
  br i1 %1130, label %1131, label %1141

1131:                                             ; preds = %1128
  %1132 = getelementptr float, ptr %1077, i64 %1121
  %1133 = mul i64 %1125, 1024
  %1134 = add i64 %1133, %1129
  %1135 = getelementptr float, ptr %1132, i64 %1134
  %1136 = load float, ptr %1135, align 4
  %1137 = getelementptr float, ptr %1112, i64 %1125
  %1138 = load float, ptr %1137, align 4
  %1139 = fadd float %1136, %1138
  store float %1139, ptr %1137, align 4
  %1140 = add i64 %1129, 1
  br label %1128

1141:                                             ; preds = %1128
  %1142 = add i64 %1125, 1
  br label %1124

1143:                                             ; preds = %1124
  %1144 = add i64 %1121, 32
  br label %1120

1145:                                             ; preds = %1120
  %1146 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %1147 = ptrtoint ptr %1146 to i64
  %1148 = add i64 %1147, 63
  %1149 = urem i64 %1148, 64
  %1150 = sub i64 %1148, %1149
  %1151 = inttoptr i64 %1150 to ptr
  br label %1152

1152:                                             ; preds = %1177, %1145
  %1153 = phi i64 [ %1178, %1177 ], [ 0, %1145 ]
  %1154 = icmp slt i64 %1153, 1024
  br i1 %1154, label %1155, label %1179

1155:                                             ; preds = %1152
  br label %1156

1156:                                             ; preds = %1175, %1155
  %1157 = phi i64 [ %1176, %1175 ], [ 0, %1155 ]
  %1158 = icmp slt i64 %1157, 1
  br i1 %1158, label %1159, label %1177

1159:                                             ; preds = %1156
  br label %1160

1160:                                             ; preds = %1163, %1159
  %1161 = phi i64 [ %1174, %1163 ], [ 0, %1159 ]
  %1162 = icmp slt i64 %1161, 32
  br i1 %1162, label %1163, label %1175

1163:                                             ; preds = %1160
  %1164 = getelementptr float, ptr %1077, i64 %1153
  %1165 = mul i64 %1157, 1024
  %1166 = add i64 %1165, %1161
  %1167 = getelementptr float, ptr %1164, i64 %1166
  %1168 = load float, ptr %1167, align 4
  %1169 = getelementptr float, ptr %1112, i64 %1157
  %1170 = load float, ptr %1169, align 4
  %1171 = fdiv float %1168, %1170
  %1172 = getelementptr float, ptr %1151, i64 %1153
  %1173 = getelementptr float, ptr %1172, i64 %1166
  store float %1171, ptr %1173, align 4
  %1174 = add i64 %1161, 1
  br label %1160

1175:                                             ; preds = %1160
  %1176 = add i64 %1157, 1
  br label %1156

1177:                                             ; preds = %1156
  %1178 = add i64 %1153, 32
  br label %1152

1179:                                             ; preds = %1152
  %1180 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64), i64 64))
  %1181 = ptrtoint ptr %1180 to i64
  %1182 = add i64 %1181, 63
  %1183 = urem i64 %1182, 64
  %1184 = sub i64 %1182, %1183
  %1185 = inttoptr i64 %1184 to ptr
  br label %1186

1186:                                             ; preds = %1205, %1179
  %1187 = phi i64 [ %1206, %1205 ], [ 0, %1179 ]
  %1188 = icmp slt i64 %1187, 64
  br i1 %1188, label %1189, label %1207

1189:                                             ; preds = %1186
  br label %1190

1190:                                             ; preds = %1203, %1189
  %1191 = phi i64 [ %1204, %1203 ], [ 0, %1189 ]
  %1192 = icmp slt i64 %1191, 1
  br i1 %1192, label %1193, label %1205

1193:                                             ; preds = %1190
  br label %1194

1194:                                             ; preds = %1197, %1193
  %1195 = phi i64 [ %1202, %1197 ], [ 0, %1193 ]
  %1196 = icmp slt i64 %1195, 32
  br i1 %1196, label %1197, label %1203

1197:                                             ; preds = %1194
  %1198 = getelementptr float, ptr %1185, i64 %1187
  %1199 = mul i64 %1191, 64
  %1200 = add i64 %1199, %1195
  %1201 = getelementptr float, ptr %1198, i64 %1200
  store float 0.000000e+00, ptr %1201, align 4
  %1202 = add i64 %1195, 1
  br label %1194

1203:                                             ; preds = %1194
  %1204 = add i64 %1191, 1
  br label %1190

1205:                                             ; preds = %1190
  %1206 = add i64 %1187, 32
  br label %1186

1207:                                             ; preds = %1186
  %1208 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64), i64 64))
  %1209 = ptrtoint ptr %1208 to i64
  %1210 = add i64 %1209, 63
  %1211 = urem i64 %1210, 64
  %1212 = sub i64 %1210, %1211
  %1213 = inttoptr i64 %1212 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1213, ptr %1185, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64), i1 false)
  br label %1214

1214:                                             ; preds = %1274, %1207
  %1215 = phi i64 [ %1275, %1274 ], [ 0, %1207 ]
  %1216 = icmp slt i64 %1215, 1024
  br i1 %1216, label %1217, label %1276

1217:                                             ; preds = %1214
  br label %1218

1218:                                             ; preds = %1272, %1217
  %1219 = phi i64 [ %1273, %1272 ], [ 0, %1217 ]
  %1220 = icmp slt i64 %1219, 64
  br i1 %1220, label %1221, label %1274

1221:                                             ; preds = %1218
  %1222 = mul i64 %1219, -1
  %1223 = add i64 %1222, 64
  %1224 = call i64 @llvm.smin.i64(i64 %1223, i64 32)
  br label %1225

1225:                                             ; preds = %1270, %1221
  %1226 = phi i64 [ %1271, %1270 ], [ 0, %1221 ]
  %1227 = icmp slt i64 %1226, 128
  br i1 %1227, label %1228, label %1272

1228:                                             ; preds = %1225
  %1229 = add i64 %1215, %1226
  %1230 = mul i64 %1215, 768
  %1231 = add i64 %782, %1230
  %1232 = mul i64 %1226, 768
  %1233 = add i64 %1231, %1232
  %1234 = add i64 %1233, %797
  %1235 = add i64 %1234, %1219
  br label %1236

1236:                                             ; preds = %1268, %1228
  %1237 = phi i64 [ %1269, %1268 ], [ 0, %1228 ]
  %1238 = icmp slt i64 %1237, 1
  br i1 %1238, label %1239, label %1270

1239:                                             ; preds = %1236
  br label %1240

1240:                                             ; preds = %1266, %1239
  %1241 = phi i64 [ %1267, %1266 ], [ 0, %1239 ]
  %1242 = icmp slt i64 %1241, %1224
  br i1 %1242, label %1243, label %1268

1243:                                             ; preds = %1240
  br label %1244

1244:                                             ; preds = %1247, %1243
  %1245 = phi i64 [ %1265, %1247 ], [ 0, %1243 ]
  %1246 = icmp slt i64 %1245, 32
  br i1 %1246, label %1247, label %1266

1247:                                             ; preds = %1244
  %1248 = getelementptr float, ptr %1151, i64 %1229
  %1249 = mul i64 %1237, 1024
  %1250 = add i64 %1249, %1245
  %1251 = getelementptr float, ptr %1248, i64 %1250
  %1252 = load float, ptr %1251, align 4
  %1253 = getelementptr float, ptr %30, i64 %1235
  %1254 = mul i64 %1245, 768
  %1255 = add i64 %1254, %1241
  %1256 = getelementptr float, ptr %1253, i64 %1255
  %1257 = load float, ptr %1256, align 4
  %1258 = getelementptr float, ptr %1213, i64 %1219
  %1259 = mul i64 %1237, 64
  %1260 = add i64 %1259, %1241
  %1261 = getelementptr float, ptr %1258, i64 %1260
  %1262 = load float, ptr %1261, align 4
  %1263 = fmul float %1252, %1257
  %1264 = fadd float %1262, %1263
  store float %1264, ptr %1261, align 4
  %1265 = add i64 %1245, 1
  br label %1244

1266:                                             ; preds = %1244
  %1267 = add i64 %1241, 1
  br label %1240

1268:                                             ; preds = %1240
  %1269 = add i64 %1237, 1
  br label %1236

1270:                                             ; preds = %1236
  %1271 = add i64 %1226, 32
  br label %1225

1272:                                             ; preds = %1225
  %1273 = add i64 %1219, 32
  br label %1218

1274:                                             ; preds = %1218
  %1275 = add i64 %1215, 128
  br label %1214

1276:                                             ; preds = %1214
  %1277 = mul i64 %794, 64
  %1278 = getelementptr float, ptr %792, i64 %1277
  call void @llvm.memcpy.p0.p0.i64(ptr %1278, ptr %1213, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64), i1 false)
  %1279 = add i64 %794, 1
  br label %793

1280:                                             ; preds = %793
  %1281 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1282 = ptrtoint ptr %1281 to i64
  %1283 = add i64 %1282, 63
  %1284 = urem i64 %1283, 64
  %1285 = sub i64 %1283, %1284
  %1286 = inttoptr i64 %1285 to ptr
  br label %1287

1287:                                             ; preds = %1306, %1280
  %1288 = phi i64 [ %1307, %1306 ], [ 0, %1280 ]
  %1289 = icmp slt i64 %1288, 768
  br i1 %1289, label %1290, label %1308

1290:                                             ; preds = %1287
  br label %1291

1291:                                             ; preds = %1304, %1290
  %1292 = phi i64 [ %1305, %1304 ], [ 0, %1290 ]
  %1293 = icmp slt i64 %1292, 1
  br i1 %1293, label %1294, label %1306

1294:                                             ; preds = %1291
  br label %1295

1295:                                             ; preds = %1298, %1294
  %1296 = phi i64 [ %1303, %1298 ], [ 0, %1294 ]
  %1297 = icmp slt i64 %1296, 32
  br i1 %1297, label %1298, label %1304

1298:                                             ; preds = %1295
  %1299 = getelementptr float, ptr %1286, i64 %1288
  %1300 = mul i64 %1292, 768
  %1301 = add i64 %1300, %1296
  %1302 = getelementptr float, ptr %1299, i64 %1301
  store float 0.000000e+00, ptr %1302, align 4
  %1303 = add i64 %1296, 1
  br label %1295

1304:                                             ; preds = %1295
  %1305 = add i64 %1292, 1
  br label %1291

1306:                                             ; preds = %1291
  %1307 = add i64 %1288, 32
  br label %1287

1308:                                             ; preds = %1287
  br label %1309

1309:                                             ; preds = %1374, %1308
  %1310 = phi i64 [ %1375, %1374 ], [ 0, %1308 ]
  %1311 = icmp slt i64 %1310, 768
  br i1 %1311, label %1312, label %1376

1312:                                             ; preds = %1309
  br label %1313

1313:                                             ; preds = %1372, %1312
  %1314 = phi i64 [ %1373, %1372 ], [ 0, %1312 ]
  %1315 = icmp slt i64 %1314, 768
  br i1 %1315, label %1316, label %1374

1316:                                             ; preds = %1313
  br label %1317

1317:                                             ; preds = %1370, %1316
  %1318 = phi i64 [ %1371, %1370 ], [ 0, %1316 ]
  %1319 = icmp slt i64 %1318, 128
  br i1 %1319, label %1320, label %1372

1320:                                             ; preds = %1317
  %1321 = add i64 %1310, %1318
  br label %1322

1322:                                             ; preds = %1368, %1320
  %1323 = phi i64 [ %1369, %1368 ], [ 0, %1320 ]
  %1324 = icmp slt i64 %1323, 128
  br i1 %1324, label %1325, label %1370

1325:                                             ; preds = %1322
  %1326 = add i64 %1314, %1323
  %1327 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %12, 1
  %1328 = mul i64 %57, 589824
  %1329 = mul i64 %1314, 768
  %1330 = add i64 %1328, %1329
  %1331 = mul i64 %1323, 768
  %1332 = add i64 %1330, %1331
  %1333 = add i64 %1332, %1310
  %1334 = add i64 %1333, %1318
  br label %1335

1335:                                             ; preds = %1366, %1325
  %1336 = phi i64 [ %1367, %1366 ], [ 0, %1325 ]
  %1337 = icmp slt i64 %1336, 1
  br i1 %1337, label %1338, label %1368

1338:                                             ; preds = %1335
  br label %1339

1339:                                             ; preds = %1364, %1338
  %1340 = phi i64 [ %1365, %1364 ], [ 0, %1338 ]
  %1341 = icmp slt i64 %1340, 32
  br i1 %1341, label %1342, label %1366

1342:                                             ; preds = %1339
  br label %1343

1343:                                             ; preds = %1346, %1342
  %1344 = phi i64 [ %1363, %1346 ], [ 0, %1342 ]
  %1345 = icmp slt i64 %1344, 32
  br i1 %1345, label %1346, label %1364

1346:                                             ; preds = %1343
  %1347 = getelementptr float, ptr %792, i64 %1326
  %1348 = mul i64 %1336, 768
  %1349 = add i64 %1348, %1344
  %1350 = getelementptr float, ptr %1347, i64 %1349
  %1351 = load float, ptr %1350, align 4
  %1352 = getelementptr float, ptr %1327, i64 %1334
  %1353 = mul i64 %1344, 768
  %1354 = add i64 %1353, %1340
  %1355 = getelementptr float, ptr %1352, i64 %1354
  %1356 = load float, ptr %1355, align 4
  %1357 = getelementptr float, ptr %1286, i64 %1321
  %1358 = add i64 %1348, %1340
  %1359 = getelementptr float, ptr %1357, i64 %1358
  %1360 = load float, ptr %1359, align 4
  %1361 = fmul float %1351, %1356
  %1362 = fadd float %1360, %1361
  store float %1362, ptr %1359, align 4
  %1363 = add i64 %1344, 1
  br label %1343

1364:                                             ; preds = %1343
  %1365 = add i64 %1340, 1
  br label %1339

1366:                                             ; preds = %1339
  %1367 = add i64 %1336, 1
  br label %1335

1368:                                             ; preds = %1335
  %1369 = add i64 %1323, 32
  br label %1322

1370:                                             ; preds = %1322
  %1371 = add i64 %1318, 32
  br label %1317

1372:                                             ; preds = %1317
  %1373 = add i64 %1314, 128
  br label %1313

1374:                                             ; preds = %1313
  %1375 = add i64 %1310, 128
  br label %1309

1376:                                             ; preds = %1309
  %1377 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1378 = ptrtoint ptr %1377 to i64
  %1379 = add i64 %1378, 63
  %1380 = urem i64 %1379, 64
  %1381 = sub i64 %1379, %1380
  %1382 = inttoptr i64 %1381 to ptr
  br label %1383

1383:                                             ; preds = %1410, %1376
  %1384 = phi i64 [ %1411, %1410 ], [ 0, %1376 ]
  %1385 = icmp slt i64 %1384, 768
  br i1 %1385, label %1386, label %1412

1386:                                             ; preds = %1383
  %1387 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  br label %1388

1388:                                             ; preds = %1408, %1386
  %1389 = phi i64 [ %1409, %1408 ], [ 0, %1386 ]
  %1390 = icmp slt i64 %1389, 1
  br i1 %1390, label %1391, label %1410

1391:                                             ; preds = %1388
  br label %1392

1392:                                             ; preds = %1395, %1391
  %1393 = phi i64 [ %1407, %1395 ], [ 0, %1391 ]
  %1394 = icmp slt i64 %1393, 32
  br i1 %1394, label %1395, label %1408

1395:                                             ; preds = %1392
  %1396 = getelementptr float, ptr %1387, i64 %1384
  %1397 = mul i64 %1389, 768
  %1398 = add i64 %1397, %1393
  %1399 = getelementptr float, ptr %1396, i64 %1398
  %1400 = load float, ptr %1399, align 4
  %1401 = getelementptr float, ptr %1286, i64 %1384
  %1402 = getelementptr float, ptr %1401, i64 %1398
  %1403 = load float, ptr %1402, align 4
  %1404 = fadd float %1400, %1403
  %1405 = getelementptr float, ptr %1382, i64 %1384
  %1406 = getelementptr float, ptr %1405, i64 %1398
  store float %1404, ptr %1406, align 4
  %1407 = add i64 %1393, 1
  br label %1392

1408:                                             ; preds = %1392
  %1409 = add i64 %1389, 1
  br label %1388

1410:                                             ; preds = %1388
  %1411 = add i64 %1384, 32
  br label %1383

1412:                                             ; preds = %1383
  %1413 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1414 = ptrtoint ptr %1413 to i64
  %1415 = add i64 %1414, 63
  %1416 = urem i64 %1415, 64
  %1417 = sub i64 %1415, %1416
  %1418 = inttoptr i64 %1417 to ptr
  br label %1419

1419:                                             ; preds = %1422, %1412
  %1420 = phi i64 [ %1424, %1422 ], [ 0, %1412 ]
  %1421 = icmp slt i64 %1420, 1
  br i1 %1421, label %1422, label %1425

1422:                                             ; preds = %1419
  %1423 = getelementptr float, ptr %1418, i64 %1420
  store float 0.000000e+00, ptr %1423, align 4
  %1424 = add i64 %1420, 1
  br label %1419

1425:                                             ; preds = %1419
  br label %1426

1426:                                             ; preds = %1457, %1425
  %1427 = phi i64 [ %1458, %1457 ], [ 0, %1425 ]
  %1428 = icmp slt i64 %1427, 768
  br i1 %1428, label %1429, label %1459

1429:                                             ; preds = %1426
  br label %1430

1430:                                             ; preds = %1455, %1429
  %1431 = phi i64 [ %1456, %1455 ], [ 0, %1429 ]
  %1432 = icmp slt i64 %1431, 128
  br i1 %1432, label %1433, label %1457

1433:                                             ; preds = %1430
  %1434 = add i64 %1427, %1431
  br label %1435

1435:                                             ; preds = %1453, %1433
  %1436 = phi i64 [ %1454, %1453 ], [ 0, %1433 ]
  %1437 = icmp slt i64 %1436, 1
  br i1 %1437, label %1438, label %1455

1438:                                             ; preds = %1435
  br label %1439

1439:                                             ; preds = %1442, %1438
  %1440 = phi i64 [ %1452, %1442 ], [ 0, %1438 ]
  %1441 = icmp slt i64 %1440, 32
  br i1 %1441, label %1442, label %1453

1442:                                             ; preds = %1439
  %1443 = getelementptr float, ptr %1382, i64 %1434
  %1444 = mul i64 %1436, 768
  %1445 = add i64 %1444, %1440
  %1446 = getelementptr float, ptr %1443, i64 %1445
  %1447 = load float, ptr %1446, align 4
  %1448 = getelementptr float, ptr %1418, i64 %1436
  %1449 = load float, ptr %1448, align 4
  %1450 = fmul float %1447, %1447
  %1451 = fadd float %1449, %1450
  store float %1451, ptr %1448, align 4
  %1452 = add i64 %1440, 1
  br label %1439

1453:                                             ; preds = %1439
  %1454 = add i64 %1436, 1
  br label %1435

1455:                                             ; preds = %1435
  %1456 = add i64 %1431, 32
  br label %1430

1457:                                             ; preds = %1430
  %1458 = add i64 %1427, 128
  br label %1426

1459:                                             ; preds = %1426
  %1460 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1461 = ptrtoint ptr %1460 to i64
  %1462 = add i64 %1461, 63
  %1463 = urem i64 %1462, 64
  %1464 = sub i64 %1462, %1463
  %1465 = inttoptr i64 %1464 to ptr
  br label %1466

1466:                                             ; preds = %1469, %1459
  %1467 = phi i64 [ %1477, %1469 ], [ 0, %1459 ]
  %1468 = icmp slt i64 %1467, 1
  br i1 %1468, label %1469, label %1478

1469:                                             ; preds = %1466
  %1470 = getelementptr float, ptr %1418, i64 %1467
  %1471 = load float, ptr %1470, align 4
  %1472 = fdiv float %1471, 7.680000e+02
  %1473 = fadd float %1472, 0x3EE4F8B580000000
  %1474 = call float @llvm.sqrt.f32(float %1473)
  %1475 = fdiv float 1.000000e+00, %1474
  %1476 = getelementptr float, ptr %1465, i64 %1467
  store float %1475, ptr %1476, align 4
  %1477 = add i64 %1467, 1
  br label %1466

1478:                                             ; preds = %1466
  %1479 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1480 = ptrtoint ptr %1479 to i64
  %1481 = add i64 %1480, 63
  %1482 = urem i64 %1481, 64
  %1483 = sub i64 %1481, %1482
  %1484 = inttoptr i64 %1483 to ptr
  br label %1485

1485:                                             ; preds = %1517, %1478
  %1486 = phi i64 [ %1518, %1517 ], [ 0, %1478 ]
  %1487 = icmp slt i64 %1486, 768
  br i1 %1487, label %1488, label %1519

1488:                                             ; preds = %1485
  %1489 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  %1490 = mul i64 %57, 768
  %1491 = add i64 %1490, %1486
  br label %1492

1492:                                             ; preds = %1515, %1488
  %1493 = phi i64 [ %1516, %1515 ], [ 0, %1488 ]
  %1494 = icmp slt i64 %1493, 1
  br i1 %1494, label %1495, label %1517

1495:                                             ; preds = %1492
  br label %1496

1496:                                             ; preds = %1499, %1495
  %1497 = phi i64 [ %1514, %1499 ], [ 0, %1495 ]
  %1498 = icmp slt i64 %1497, 32
  br i1 %1498, label %1499, label %1515

1499:                                             ; preds = %1496
  %1500 = getelementptr float, ptr %1382, i64 %1486
  %1501 = mul i64 %1493, 768
  %1502 = add i64 %1501, %1497
  %1503 = getelementptr float, ptr %1500, i64 %1502
  %1504 = load float, ptr %1503, align 4
  %1505 = getelementptr float, ptr %1465, i64 %1493
  %1506 = load float, ptr %1505, align 4
  %1507 = getelementptr float, ptr %1489, i64 %1491
  %1508 = getelementptr float, ptr %1507, i64 %1497
  %1509 = load float, ptr %1508, align 4
  %1510 = fmul float %1504, %1506
  %1511 = fmul float %1510, %1509
  %1512 = getelementptr float, ptr %1484, i64 %1486
  %1513 = getelementptr float, ptr %1512, i64 %1502
  store float %1511, ptr %1513, align 4
  %1514 = add i64 %1497, 1
  br label %1496

1515:                                             ; preds = %1496
  %1516 = add i64 %1493, 1
  br label %1492

1517:                                             ; preds = %1492
  %1518 = add i64 %1486, 32
  br label %1485

1519:                                             ; preds = %1485
  %1520 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %1521 = ptrtoint ptr %1520 to i64
  %1522 = add i64 %1521, 63
  %1523 = urem i64 %1522, 64
  %1524 = sub i64 %1522, %1523
  %1525 = inttoptr i64 %1524 to ptr
  br label %1526

1526:                                             ; preds = %1545, %1519
  %1527 = phi i64 [ %1546, %1545 ], [ 0, %1519 ]
  %1528 = icmp slt i64 %1527, 2048
  br i1 %1528, label %1529, label %1547

1529:                                             ; preds = %1526
  br label %1530

1530:                                             ; preds = %1543, %1529
  %1531 = phi i64 [ %1544, %1543 ], [ 0, %1529 ]
  %1532 = icmp slt i64 %1531, 1
  br i1 %1532, label %1533, label %1545

1533:                                             ; preds = %1530
  br label %1534

1534:                                             ; preds = %1537, %1533
  %1535 = phi i64 [ %1542, %1537 ], [ 0, %1533 ]
  %1536 = icmp slt i64 %1535, 32
  br i1 %1536, label %1537, label %1543

1537:                                             ; preds = %1534
  %1538 = getelementptr float, ptr %1525, i64 %1527
  %1539 = mul i64 %1531, 2048
  %1540 = add i64 %1539, %1535
  %1541 = getelementptr float, ptr %1538, i64 %1540
  store float 0.000000e+00, ptr %1541, align 4
  %1542 = add i64 %1535, 1
  br label %1534

1543:                                             ; preds = %1534
  %1544 = add i64 %1531, 1
  br label %1530

1545:                                             ; preds = %1530
  %1546 = add i64 %1527, 32
  br label %1526

1547:                                             ; preds = %1526
  br label %1548

1548:                                             ; preds = %1614, %1547
  %1549 = phi i64 [ %1615, %1614 ], [ 0, %1547 ]
  %1550 = icmp slt i64 %1549, 2048
  br i1 %1550, label %1551, label %1616

1551:                                             ; preds = %1548
  br label %1552

1552:                                             ; preds = %1612, %1551
  %1553 = phi i64 [ %1613, %1612 ], [ 0, %1551 ]
  %1554 = icmp slt i64 %1553, 768
  br i1 %1554, label %1555, label %1614

1555:                                             ; preds = %1552
  br label %1556

1556:                                             ; preds = %1610, %1555
  %1557 = phi i64 [ %1611, %1610 ], [ 0, %1555 ]
  %1558 = icmp slt i64 %1557, 128
  br i1 %1558, label %1559, label %1612

1559:                                             ; preds = %1556
  %1560 = add i64 %1549, %1557
  br label %1561

1561:                                             ; preds = %1608, %1559
  %1562 = phi i64 [ %1609, %1608 ], [ 0, %1559 ]
  %1563 = icmp slt i64 %1562, 128
  br i1 %1563, label %1564, label %1610

1564:                                             ; preds = %1561
  %1565 = add i64 %1553, %1562
  %1566 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 1
  %1567 = mul i64 %57, 1572864
  %1568 = mul i64 %1553, 2048
  %1569 = add i64 %1567, %1568
  %1570 = mul i64 %1562, 2048
  %1571 = add i64 %1569, %1570
  %1572 = add i64 %1571, %1549
  %1573 = add i64 %1572, %1557
  br label %1574

1574:                                             ; preds = %1606, %1564
  %1575 = phi i64 [ %1607, %1606 ], [ 0, %1564 ]
  %1576 = icmp slt i64 %1575, 1
  br i1 %1576, label %1577, label %1608

1577:                                             ; preds = %1574
  br label %1578

1578:                                             ; preds = %1604, %1577
  %1579 = phi i64 [ %1605, %1604 ], [ 0, %1577 ]
  %1580 = icmp slt i64 %1579, 32
  br i1 %1580, label %1581, label %1606

1581:                                             ; preds = %1578
  br label %1582

1582:                                             ; preds = %1585, %1581
  %1583 = phi i64 [ %1603, %1585 ], [ 0, %1581 ]
  %1584 = icmp slt i64 %1583, 32
  br i1 %1584, label %1585, label %1604

1585:                                             ; preds = %1582
  %1586 = getelementptr float, ptr %1484, i64 %1565
  %1587 = mul i64 %1575, 768
  %1588 = add i64 %1587, %1583
  %1589 = getelementptr float, ptr %1586, i64 %1588
  %1590 = load float, ptr %1589, align 4
  %1591 = getelementptr float, ptr %1566, i64 %1573
  %1592 = mul i64 %1583, 2048
  %1593 = add i64 %1592, %1579
  %1594 = getelementptr float, ptr %1591, i64 %1593
  %1595 = load float, ptr %1594, align 4
  %1596 = getelementptr float, ptr %1525, i64 %1560
  %1597 = mul i64 %1575, 2048
  %1598 = add i64 %1597, %1579
  %1599 = getelementptr float, ptr %1596, i64 %1598
  %1600 = load float, ptr %1599, align 4
  %1601 = fmul float %1590, %1595
  %1602 = fadd float %1600, %1601
  store float %1602, ptr %1599, align 4
  %1603 = add i64 %1583, 1
  br label %1582

1604:                                             ; preds = %1582
  %1605 = add i64 %1579, 1
  br label %1578

1606:                                             ; preds = %1578
  %1607 = add i64 %1575, 1
  br label %1574

1608:                                             ; preds = %1574
  %1609 = add i64 %1562, 32
  br label %1561

1610:                                             ; preds = %1561
  %1611 = add i64 %1557, 32
  br label %1556

1612:                                             ; preds = %1556
  %1613 = add i64 %1553, 128
  br label %1552

1614:                                             ; preds = %1552
  %1615 = add i64 %1549, 128
  br label %1548

1616:                                             ; preds = %1548
  %1617 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %1618 = ptrtoint ptr %1617 to i64
  %1619 = add i64 %1618, 63
  %1620 = urem i64 %1619, 64
  %1621 = sub i64 %1619, %1620
  %1622 = inttoptr i64 %1621 to ptr
  br label %1623

1623:                                             ; preds = %1642, %1616
  %1624 = phi i64 [ %1643, %1642 ], [ 0, %1616 ]
  %1625 = icmp slt i64 %1624, 2048
  br i1 %1625, label %1626, label %1644

1626:                                             ; preds = %1623
  br label %1627

1627:                                             ; preds = %1640, %1626
  %1628 = phi i64 [ %1641, %1640 ], [ 0, %1626 ]
  %1629 = icmp slt i64 %1628, 1
  br i1 %1629, label %1630, label %1642

1630:                                             ; preds = %1627
  br label %1631

1631:                                             ; preds = %1634, %1630
  %1632 = phi i64 [ %1639, %1634 ], [ 0, %1630 ]
  %1633 = icmp slt i64 %1632, 32
  br i1 %1633, label %1634, label %1640

1634:                                             ; preds = %1631
  %1635 = getelementptr float, ptr %1622, i64 %1624
  %1636 = mul i64 %1628, 2048
  %1637 = add i64 %1636, %1632
  %1638 = getelementptr float, ptr %1635, i64 %1637
  store float 0.000000e+00, ptr %1638, align 4
  %1639 = add i64 %1632, 1
  br label %1631

1640:                                             ; preds = %1631
  %1641 = add i64 %1628, 1
  br label %1627

1642:                                             ; preds = %1627
  %1643 = add i64 %1624, 32
  br label %1623

1644:                                             ; preds = %1623
  br label %1645

1645:                                             ; preds = %1711, %1644
  %1646 = phi i64 [ %1712, %1711 ], [ 0, %1644 ]
  %1647 = icmp slt i64 %1646, 2048
  br i1 %1647, label %1648, label %1713

1648:                                             ; preds = %1645
  br label %1649

1649:                                             ; preds = %1709, %1648
  %1650 = phi i64 [ %1710, %1709 ], [ 0, %1648 ]
  %1651 = icmp slt i64 %1650, 768
  br i1 %1651, label %1652, label %1711

1652:                                             ; preds = %1649
  br label %1653

1653:                                             ; preds = %1707, %1652
  %1654 = phi i64 [ %1708, %1707 ], [ 0, %1652 ]
  %1655 = icmp slt i64 %1654, 128
  br i1 %1655, label %1656, label %1709

1656:                                             ; preds = %1653
  %1657 = add i64 %1646, %1654
  br label %1658

1658:                                             ; preds = %1705, %1656
  %1659 = phi i64 [ %1706, %1705 ], [ 0, %1656 ]
  %1660 = icmp slt i64 %1659, 128
  br i1 %1660, label %1661, label %1707

1661:                                             ; preds = %1658
  %1662 = add i64 %1650, %1659
  %1663 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %16, 1
  %1664 = mul i64 %57, 1572864
  %1665 = mul i64 %1650, 2048
  %1666 = add i64 %1664, %1665
  %1667 = mul i64 %1659, 2048
  %1668 = add i64 %1666, %1667
  %1669 = add i64 %1668, %1646
  %1670 = add i64 %1669, %1654
  br label %1671

1671:                                             ; preds = %1703, %1661
  %1672 = phi i64 [ %1704, %1703 ], [ 0, %1661 ]
  %1673 = icmp slt i64 %1672, 1
  br i1 %1673, label %1674, label %1705

1674:                                             ; preds = %1671
  br label %1675

1675:                                             ; preds = %1701, %1674
  %1676 = phi i64 [ %1702, %1701 ], [ 0, %1674 ]
  %1677 = icmp slt i64 %1676, 32
  br i1 %1677, label %1678, label %1703

1678:                                             ; preds = %1675
  br label %1679

1679:                                             ; preds = %1682, %1678
  %1680 = phi i64 [ %1700, %1682 ], [ 0, %1678 ]
  %1681 = icmp slt i64 %1680, 32
  br i1 %1681, label %1682, label %1701

1682:                                             ; preds = %1679
  %1683 = getelementptr float, ptr %1484, i64 %1662
  %1684 = mul i64 %1672, 768
  %1685 = add i64 %1684, %1680
  %1686 = getelementptr float, ptr %1683, i64 %1685
  %1687 = load float, ptr %1686, align 4
  %1688 = getelementptr float, ptr %1663, i64 %1670
  %1689 = mul i64 %1680, 2048
  %1690 = add i64 %1689, %1676
  %1691 = getelementptr float, ptr %1688, i64 %1690
  %1692 = load float, ptr %1691, align 4
  %1693 = getelementptr float, ptr %1622, i64 %1657
  %1694 = mul i64 %1672, 2048
  %1695 = add i64 %1694, %1676
  %1696 = getelementptr float, ptr %1693, i64 %1695
  %1697 = load float, ptr %1696, align 4
  %1698 = fmul float %1687, %1692
  %1699 = fadd float %1697, %1698
  store float %1699, ptr %1696, align 4
  %1700 = add i64 %1680, 1
  br label %1679

1701:                                             ; preds = %1679
  %1702 = add i64 %1676, 1
  br label %1675

1703:                                             ; preds = %1675
  %1704 = add i64 %1672, 1
  br label %1671

1705:                                             ; preds = %1671
  %1706 = add i64 %1659, 32
  br label %1658

1707:                                             ; preds = %1658
  %1708 = add i64 %1654, 32
  br label %1653

1709:                                             ; preds = %1653
  %1710 = add i64 %1650, 128
  br label %1649

1711:                                             ; preds = %1649
  %1712 = add i64 %1646, 128
  br label %1645

1713:                                             ; preds = %1645
  %1714 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %1715 = ptrtoint ptr %1714 to i64
  %1716 = add i64 %1715, 63
  %1717 = urem i64 %1716, 64
  %1718 = sub i64 %1716, %1717
  %1719 = inttoptr i64 %1718 to ptr
  br label %1720

1720:                                             ; preds = %1746, %1713
  %1721 = phi i64 [ %1747, %1746 ], [ 0, %1713 ]
  %1722 = icmp slt i64 %1721, 2048
  br i1 %1722, label %1723, label %1748

1723:                                             ; preds = %1720
  br label %1724

1724:                                             ; preds = %1744, %1723
  %1725 = phi i64 [ %1745, %1744 ], [ 0, %1723 ]
  %1726 = icmp slt i64 %1725, 1
  br i1 %1726, label %1727, label %1746

1727:                                             ; preds = %1724
  br label %1728

1728:                                             ; preds = %1731, %1727
  %1729 = phi i64 [ %1743, %1731 ], [ 0, %1727 ]
  %1730 = icmp slt i64 %1729, 32
  br i1 %1730, label %1731, label %1744

1731:                                             ; preds = %1728
  %1732 = getelementptr float, ptr %1525, i64 %1721
  %1733 = mul i64 %1725, 2048
  %1734 = add i64 %1733, %1729
  %1735 = getelementptr float, ptr %1732, i64 %1734
  %1736 = load float, ptr %1735, align 4
  %1737 = fneg float %1736
  %1738 = call float @llvm.exp.f32(float %1737)
  %1739 = fadd float %1738, 1.000000e+00
  %1740 = fdiv float %1736, %1739
  %1741 = getelementptr float, ptr %1719, i64 %1721
  %1742 = getelementptr float, ptr %1741, i64 %1734
  store float %1740, ptr %1742, align 4
  %1743 = add i64 %1729, 1
  br label %1728

1744:                                             ; preds = %1728
  %1745 = add i64 %1725, 1
  br label %1724

1746:                                             ; preds = %1724
  %1747 = add i64 %1721, 32
  br label %1720

1748:                                             ; preds = %1720
  %1749 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %1750 = ptrtoint ptr %1749 to i64
  %1751 = add i64 %1750, 63
  %1752 = urem i64 %1751, 64
  %1753 = sub i64 %1751, %1752
  %1754 = inttoptr i64 %1753 to ptr
  br label %1755

1755:                                             ; preds = %1781, %1748
  %1756 = phi i64 [ %1782, %1781 ], [ 0, %1748 ]
  %1757 = icmp slt i64 %1756, 2048
  br i1 %1757, label %1758, label %1783

1758:                                             ; preds = %1755
  br label %1759

1759:                                             ; preds = %1779, %1758
  %1760 = phi i64 [ %1780, %1779 ], [ 0, %1758 ]
  %1761 = icmp slt i64 %1760, 1
  br i1 %1761, label %1762, label %1781

1762:                                             ; preds = %1759
  br label %1763

1763:                                             ; preds = %1766, %1762
  %1764 = phi i64 [ %1778, %1766 ], [ 0, %1762 ]
  %1765 = icmp slt i64 %1764, 32
  br i1 %1765, label %1766, label %1779

1766:                                             ; preds = %1763
  %1767 = getelementptr float, ptr %1719, i64 %1756
  %1768 = mul i64 %1760, 2048
  %1769 = add i64 %1768, %1764
  %1770 = getelementptr float, ptr %1767, i64 %1769
  %1771 = load float, ptr %1770, align 4
  %1772 = getelementptr float, ptr %1622, i64 %1756
  %1773 = getelementptr float, ptr %1772, i64 %1769
  %1774 = load float, ptr %1773, align 4
  %1775 = fmul float %1771, %1774
  %1776 = getelementptr float, ptr %1754, i64 %1756
  %1777 = getelementptr float, ptr %1776, i64 %1769
  store float %1775, ptr %1777, align 4
  %1778 = add i64 %1764, 1
  br label %1763

1779:                                             ; preds = %1763
  %1780 = add i64 %1760, 1
  br label %1759

1781:                                             ; preds = %1759
  %1782 = add i64 %1756, 32
  br label %1755

1783:                                             ; preds = %1755
  %1784 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1785 = ptrtoint ptr %1784 to i64
  %1786 = add i64 %1785, 63
  %1787 = urem i64 %1786, 64
  %1788 = sub i64 %1786, %1787
  %1789 = inttoptr i64 %1788 to ptr
  br label %1790

1790:                                             ; preds = %1809, %1783
  %1791 = phi i64 [ %1810, %1809 ], [ 0, %1783 ]
  %1792 = icmp slt i64 %1791, 768
  br i1 %1792, label %1793, label %1811

1793:                                             ; preds = %1790
  br label %1794

1794:                                             ; preds = %1807, %1793
  %1795 = phi i64 [ %1808, %1807 ], [ 0, %1793 ]
  %1796 = icmp slt i64 %1795, 1
  br i1 %1796, label %1797, label %1809

1797:                                             ; preds = %1794
  br label %1798

1798:                                             ; preds = %1801, %1797
  %1799 = phi i64 [ %1806, %1801 ], [ 0, %1797 ]
  %1800 = icmp slt i64 %1799, 32
  br i1 %1800, label %1801, label %1807

1801:                                             ; preds = %1798
  %1802 = getelementptr float, ptr %1789, i64 %1791
  %1803 = mul i64 %1795, 768
  %1804 = add i64 %1803, %1799
  %1805 = getelementptr float, ptr %1802, i64 %1804
  store float 0.000000e+00, ptr %1805, align 4
  %1806 = add i64 %1799, 1
  br label %1798

1807:                                             ; preds = %1798
  %1808 = add i64 %1795, 1
  br label %1794

1809:                                             ; preds = %1794
  %1810 = add i64 %1791, 32
  br label %1790

1811:                                             ; preds = %1790
  br label %1812

1812:                                             ; preds = %1878, %1811
  %1813 = phi i64 [ %1879, %1878 ], [ 0, %1811 ]
  %1814 = icmp slt i64 %1813, 768
  br i1 %1814, label %1815, label %1880

1815:                                             ; preds = %1812
  br label %1816

1816:                                             ; preds = %1876, %1815
  %1817 = phi i64 [ %1877, %1876 ], [ 0, %1815 ]
  %1818 = icmp slt i64 %1817, 2048
  br i1 %1818, label %1819, label %1878

1819:                                             ; preds = %1816
  br label %1820

1820:                                             ; preds = %1874, %1819
  %1821 = phi i64 [ %1875, %1874 ], [ 0, %1819 ]
  %1822 = icmp slt i64 %1821, 128
  br i1 %1822, label %1823, label %1876

1823:                                             ; preds = %1820
  %1824 = add i64 %1813, %1821
  br label %1825

1825:                                             ; preds = %1872, %1823
  %1826 = phi i64 [ %1873, %1872 ], [ 0, %1823 ]
  %1827 = icmp slt i64 %1826, 128
  br i1 %1827, label %1828, label %1874

1828:                                             ; preds = %1825
  %1829 = add i64 %1817, %1826
  %1830 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %15, 1
  %1831 = mul i64 %57, 1572864
  %1832 = mul i64 %1817, 768
  %1833 = add i64 %1831, %1832
  %1834 = mul i64 %1826, 768
  %1835 = add i64 %1833, %1834
  %1836 = add i64 %1835, %1813
  %1837 = add i64 %1836, %1821
  br label %1838

1838:                                             ; preds = %1870, %1828
  %1839 = phi i64 [ %1871, %1870 ], [ 0, %1828 ]
  %1840 = icmp slt i64 %1839, 1
  br i1 %1840, label %1841, label %1872

1841:                                             ; preds = %1838
  br label %1842

1842:                                             ; preds = %1868, %1841
  %1843 = phi i64 [ %1869, %1868 ], [ 0, %1841 ]
  %1844 = icmp slt i64 %1843, 32
  br i1 %1844, label %1845, label %1870

1845:                                             ; preds = %1842
  br label %1846

1846:                                             ; preds = %1849, %1845
  %1847 = phi i64 [ %1867, %1849 ], [ 0, %1845 ]
  %1848 = icmp slt i64 %1847, 32
  br i1 %1848, label %1849, label %1868

1849:                                             ; preds = %1846
  %1850 = getelementptr float, ptr %1754, i64 %1829
  %1851 = mul i64 %1839, 2048
  %1852 = add i64 %1851, %1847
  %1853 = getelementptr float, ptr %1850, i64 %1852
  %1854 = load float, ptr %1853, align 4
  %1855 = getelementptr float, ptr %1830, i64 %1837
  %1856 = mul i64 %1847, 768
  %1857 = add i64 %1856, %1843
  %1858 = getelementptr float, ptr %1855, i64 %1857
  %1859 = load float, ptr %1858, align 4
  %1860 = getelementptr float, ptr %1789, i64 %1824
  %1861 = mul i64 %1839, 768
  %1862 = add i64 %1861, %1843
  %1863 = getelementptr float, ptr %1860, i64 %1862
  %1864 = load float, ptr %1863, align 4
  %1865 = fmul float %1854, %1859
  %1866 = fadd float %1864, %1865
  store float %1866, ptr %1863, align 4
  %1867 = add i64 %1847, 1
  br label %1846

1868:                                             ; preds = %1846
  %1869 = add i64 %1843, 1
  br label %1842

1870:                                             ; preds = %1842
  %1871 = add i64 %1839, 1
  br label %1838

1872:                                             ; preds = %1838
  %1873 = add i64 %1826, 32
  br label %1825

1874:                                             ; preds = %1825
  %1875 = add i64 %1821, 32
  br label %1820

1876:                                             ; preds = %1820
  %1877 = add i64 %1817, 128
  br label %1816

1878:                                             ; preds = %1816
  %1879 = add i64 %1813, 128
  br label %1812

1880:                                             ; preds = %1812
  %1881 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1882 = ptrtoint ptr %1881 to i64
  %1883 = add i64 %1882, 63
  %1884 = urem i64 %1883, 64
  %1885 = sub i64 %1883, %1884
  %1886 = inttoptr i64 %1885 to ptr
  %1887 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1881, 0
  %1888 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1887, ptr %1886, 1
  %1889 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1888, i64 0, 2
  %1890 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1889, i64 1, 3, 0
  %1891 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1890, i64 768, 3, 1
  %1892 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1891, i64 768, 4, 0
  %1893 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1892, i64 1, 4, 1
  br label %1894

1894:                                             ; preds = %1920, %1880
  %1895 = phi i64 [ %1921, %1920 ], [ 0, %1880 ]
  %1896 = icmp slt i64 %1895, 768
  br i1 %1896, label %1897, label %1922

1897:                                             ; preds = %1894
  br label %1898

1898:                                             ; preds = %1918, %1897
  %1899 = phi i64 [ %1919, %1918 ], [ 0, %1897 ]
  %1900 = icmp slt i64 %1899, 1
  br i1 %1900, label %1901, label %1920

1901:                                             ; preds = %1898
  br label %1902

1902:                                             ; preds = %1905, %1901
  %1903 = phi i64 [ %1917, %1905 ], [ 0, %1901 ]
  %1904 = icmp slt i64 %1903, 32
  br i1 %1904, label %1905, label %1918

1905:                                             ; preds = %1902
  %1906 = getelementptr float, ptr %1382, i64 %1895
  %1907 = mul i64 %1899, 768
  %1908 = add i64 %1907, %1903
  %1909 = getelementptr float, ptr %1906, i64 %1908
  %1910 = load float, ptr %1909, align 4
  %1911 = getelementptr float, ptr %1789, i64 %1895
  %1912 = getelementptr float, ptr %1911, i64 %1908
  %1913 = load float, ptr %1912, align 4
  %1914 = fadd float %1910, %1913
  %1915 = getelementptr float, ptr %1886, i64 %1895
  %1916 = getelementptr float, ptr %1915, i64 %1908
  store float %1914, ptr %1916, align 4
  %1917 = add i64 %1903, 1
  br label %1902

1918:                                             ; preds = %1902
  %1919 = add i64 %1899, 1
  br label %1898

1920:                                             ; preds = %1898
  %1921 = add i64 %1895, 32
  br label %1894

1922:                                             ; preds = %1894
  %1923 = add i64 %57, 1
  br label %56

1924:                                             ; preds = %56
  %1925 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1926 = ptrtoint ptr %1925 to i64
  %1927 = add i64 %1926, 63
  %1928 = urem i64 %1927, 64
  %1929 = sub i64 %1927, %1928
  %1930 = inttoptr i64 %1929 to ptr
  br label %1931

1931:                                             ; preds = %1934, %1924
  %1932 = phi i64 [ %1936, %1934 ], [ 0, %1924 ]
  %1933 = icmp slt i64 %1932, 1
  br i1 %1933, label %1934, label %1937

1934:                                             ; preds = %1931
  %1935 = getelementptr float, ptr %1930, i64 %1932
  store float 0.000000e+00, ptr %1935, align 4
  %1936 = add i64 %1932, 1
  br label %1931

1937:                                             ; preds = %1931
  br label %1938

1938:                                             ; preds = %1970, %1937
  %1939 = phi i64 [ %1971, %1970 ], [ 0, %1937 ]
  %1940 = icmp slt i64 %1939, 768
  br i1 %1940, label %1941, label %1972

1941:                                             ; preds = %1938
  br label %1942

1942:                                             ; preds = %1968, %1941
  %1943 = phi i64 [ %1969, %1968 ], [ 0, %1941 ]
  %1944 = icmp slt i64 %1943, 128
  br i1 %1944, label %1945, label %1970

1945:                                             ; preds = %1942
  %1946 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %1947 = add i64 %1939, %1943
  br label %1948

1948:                                             ; preds = %1966, %1945
  %1949 = phi i64 [ %1967, %1966 ], [ 0, %1945 ]
  %1950 = icmp slt i64 %1949, 1
  br i1 %1950, label %1951, label %1968

1951:                                             ; preds = %1948
  br label %1952

1952:                                             ; preds = %1955, %1951
  %1953 = phi i64 [ %1965, %1955 ], [ 0, %1951 ]
  %1954 = icmp slt i64 %1953, 32
  br i1 %1954, label %1955, label %1966

1955:                                             ; preds = %1952
  %1956 = getelementptr float, ptr %1946, i64 %1947
  %1957 = mul i64 %1949, 768
  %1958 = add i64 %1957, %1953
  %1959 = getelementptr float, ptr %1956, i64 %1958
  %1960 = load float, ptr %1959, align 4
  %1961 = getelementptr float, ptr %1930, i64 %1949
  %1962 = load float, ptr %1961, align 4
  %1963 = fmul float %1960, %1960
  %1964 = fadd float %1962, %1963
  store float %1964, ptr %1961, align 4
  %1965 = add i64 %1953, 1
  br label %1952

1966:                                             ; preds = %1952
  %1967 = add i64 %1949, 1
  br label %1948

1968:                                             ; preds = %1948
  %1969 = add i64 %1943, 32
  br label %1942

1970:                                             ; preds = %1942
  %1971 = add i64 %1939, 128
  br label %1938

1972:                                             ; preds = %1938
  %1973 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1974 = ptrtoint ptr %1973 to i64
  %1975 = add i64 %1974, 63
  %1976 = urem i64 %1975, 64
  %1977 = sub i64 %1975, %1976
  %1978 = inttoptr i64 %1977 to ptr
  br label %1979

1979:                                             ; preds = %1982, %1972
  %1980 = phi i64 [ %1990, %1982 ], [ 0, %1972 ]
  %1981 = icmp slt i64 %1980, 1
  br i1 %1981, label %1982, label %1991

1982:                                             ; preds = %1979
  %1983 = getelementptr float, ptr %1930, i64 %1980
  %1984 = load float, ptr %1983, align 4
  %1985 = fdiv float %1984, 7.680000e+02
  %1986 = fadd float %1985, 0x3EE4F8B580000000
  %1987 = call float @llvm.sqrt.f32(float %1986)
  %1988 = fdiv float 1.000000e+00, %1987
  %1989 = getelementptr float, ptr %1978, i64 %1980
  store float %1988, ptr %1989, align 4
  %1990 = add i64 %1980, 1
  br label %1979

1991:                                             ; preds = %1979
  %1992 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1993 = ptrtoint ptr %1992 to i64
  %1994 = add i64 %1993, 63
  %1995 = urem i64 %1994, 64
  %1996 = sub i64 %1994, %1995
  %1997 = inttoptr i64 %1996 to ptr
  br label %1998

1998:                                             ; preds = %2029, %1991
  %1999 = phi i64 [ %2030, %2029 ], [ 0, %1991 ]
  %2000 = icmp slt i64 %1999, 768
  br i1 %2000, label %2001, label %2031

2001:                                             ; preds = %1998
  %2002 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %2003 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 1
  br label %2004

2004:                                             ; preds = %2027, %2001
  %2005 = phi i64 [ %2028, %2027 ], [ 0, %2001 ]
  %2006 = icmp slt i64 %2005, 1
  br i1 %2006, label %2007, label %2029

2007:                                             ; preds = %2004
  br label %2008

2008:                                             ; preds = %2011, %2007
  %2009 = phi i64 [ %2026, %2011 ], [ 0, %2007 ]
  %2010 = icmp slt i64 %2009, 32
  br i1 %2010, label %2011, label %2027

2011:                                             ; preds = %2008
  %2012 = getelementptr float, ptr %2002, i64 %1999
  %2013 = mul i64 %2005, 768
  %2014 = add i64 %2013, %2009
  %2015 = getelementptr float, ptr %2012, i64 %2014
  %2016 = load float, ptr %2015, align 4
  %2017 = getelementptr float, ptr %1978, i64 %2005
  %2018 = load float, ptr %2017, align 4
  %2019 = getelementptr float, ptr %2003, i64 %1999
  %2020 = getelementptr float, ptr %2019, i64 %2009
  %2021 = load float, ptr %2020, align 4
  %2022 = fmul float %2016, %2018
  %2023 = fmul float %2022, %2021
  %2024 = getelementptr float, ptr %1997, i64 %1999
  %2025 = getelementptr float, ptr %2024, i64 %2014
  store float %2023, ptr %2025, align 4
  %2026 = add i64 %2009, 1
  br label %2008

2027:                                             ; preds = %2008
  %2028 = add i64 %2005, 1
  br label %2004

2029:                                             ; preds = %2004
  %2030 = add i64 %1999, 32
  br label %1998

2031:                                             ; preds = %1998
  %2032 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32000) to i64), i64 64))
  %2033 = ptrtoint ptr %2032 to i64
  %2034 = add i64 %2033, 63
  %2035 = urem i64 %2034, 64
  %2036 = sub i64 %2034, %2035
  %2037 = inttoptr i64 %2036 to ptr
  br label %2038

2038:                                             ; preds = %2057, %2031
  %2039 = phi i64 [ %2058, %2057 ], [ 0, %2031 ]
  %2040 = icmp slt i64 %2039, 32000
  br i1 %2040, label %2041, label %2059

2041:                                             ; preds = %2038
  br label %2042

2042:                                             ; preds = %2055, %2041
  %2043 = phi i64 [ %2056, %2055 ], [ 0, %2041 ]
  %2044 = icmp slt i64 %2043, 1
  br i1 %2044, label %2045, label %2057

2045:                                             ; preds = %2042
  br label %2046

2046:                                             ; preds = %2049, %2045
  %2047 = phi i64 [ %2054, %2049 ], [ 0, %2045 ]
  %2048 = icmp slt i64 %2047, 32
  br i1 %2048, label %2049, label %2055

2049:                                             ; preds = %2046
  %2050 = getelementptr float, ptr %2037, i64 %2039
  %2051 = mul i64 %2043, 32000
  %2052 = add i64 %2051, %2047
  %2053 = getelementptr float, ptr %2050, i64 %2052
  store float 0.000000e+00, ptr %2053, align 4
  %2054 = add i64 %2047, 1
  br label %2046

2055:                                             ; preds = %2046
  %2056 = add i64 %2043, 1
  br label %2042

2057:                                             ; preds = %2042
  %2058 = add i64 %2039, 32
  br label %2038

2059:                                             ; preds = %2038
  br label %2060

2060:                                             ; preds = %2124, %2059
  %2061 = phi i64 [ %2125, %2124 ], [ 0, %2059 ]
  %2062 = icmp slt i64 %2061, 32000
  br i1 %2062, label %2063, label %2126

2063:                                             ; preds = %2060
  br label %2064

2064:                                             ; preds = %2122, %2063
  %2065 = phi i64 [ %2123, %2122 ], [ 0, %2063 ]
  %2066 = icmp slt i64 %2065, 768
  br i1 %2066, label %2067, label %2124

2067:                                             ; preds = %2064
  br label %2068

2068:                                             ; preds = %2120, %2067
  %2069 = phi i64 [ %2121, %2120 ], [ 0, %2067 ]
  %2070 = icmp slt i64 %2069, 128
  br i1 %2070, label %2071, label %2122

2071:                                             ; preds = %2068
  %2072 = add i64 %2061, %2069
  br label %2073

2073:                                             ; preds = %2118, %2071
  %2074 = phi i64 [ %2119, %2118 ], [ 0, %2071 ]
  %2075 = icmp slt i64 %2074, 128
  br i1 %2075, label %2076, label %2120

2076:                                             ; preds = %2073
  %2077 = add i64 %2065, %2074
  %2078 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %2079 = mul i64 %2065, 32000
  %2080 = mul i64 %2074, 32000
  %2081 = add i64 %2079, %2080
  %2082 = add i64 %2081, %2061
  %2083 = add i64 %2082, %2069
  br label %2084

2084:                                             ; preds = %2116, %2076
  %2085 = phi i64 [ %2117, %2116 ], [ 0, %2076 ]
  %2086 = icmp slt i64 %2085, 1
  br i1 %2086, label %2087, label %2118

2087:                                             ; preds = %2084
  br label %2088

2088:                                             ; preds = %2114, %2087
  %2089 = phi i64 [ %2115, %2114 ], [ 0, %2087 ]
  %2090 = icmp slt i64 %2089, 32
  br i1 %2090, label %2091, label %2116

2091:                                             ; preds = %2088
  br label %2092

2092:                                             ; preds = %2095, %2091
  %2093 = phi i64 [ %2113, %2095 ], [ 0, %2091 ]
  %2094 = icmp slt i64 %2093, 32
  br i1 %2094, label %2095, label %2114

2095:                                             ; preds = %2092
  %2096 = getelementptr float, ptr %1997, i64 %2077
  %2097 = mul i64 %2085, 768
  %2098 = add i64 %2097, %2093
  %2099 = getelementptr float, ptr %2096, i64 %2098
  %2100 = load float, ptr %2099, align 4
  %2101 = getelementptr float, ptr %2078, i64 %2083
  %2102 = mul i64 %2093, 32000
  %2103 = add i64 %2102, %2089
  %2104 = getelementptr float, ptr %2101, i64 %2103
  %2105 = load float, ptr %2104, align 4
  %2106 = getelementptr float, ptr %2037, i64 %2072
  %2107 = mul i64 %2085, 32000
  %2108 = add i64 %2107, %2089
  %2109 = getelementptr float, ptr %2106, i64 %2108
  %2110 = load float, ptr %2109, align 4
  %2111 = fmul float %2100, %2105
  %2112 = fadd float %2110, %2111
  store float %2112, ptr %2109, align 4
  %2113 = add i64 %2093, 1
  br label %2092

2114:                                             ; preds = %2092
  %2115 = add i64 %2089, 1
  br label %2088

2116:                                             ; preds = %2088
  %2117 = add i64 %2085, 1
  br label %2084

2118:                                             ; preds = %2084
  %2119 = add i64 %2074, 32
  br label %2073

2120:                                             ; preds = %2073
  %2121 = add i64 %2069, 32
  br label %2068

2122:                                             ; preds = %2068
  %2123 = add i64 %2065, 128
  br label %2064

2124:                                             ; preds = %2064
  %2125 = add i64 %2061, 128
  br label %2060

2126:                                             ; preds = %2060
  %2127 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %2128 = ptrtoint ptr %2127 to i64
  %2129 = add i64 %2128, 63
  %2130 = urem i64 %2129, 64
  %2131 = sub i64 %2129, %2130
  %2132 = inttoptr i64 %2131 to ptr
  br label %2133

2133:                                             ; preds = %2136, %2126
  %2134 = phi i64 [ %2138, %2136 ], [ 0, %2126 ]
  %2135 = icmp slt i64 %2134, 1
  br i1 %2135, label %2136, label %2139

2136:                                             ; preds = %2133
  %2137 = getelementptr float, ptr %2132, i64 %2134
  store float 0xFFF0000000000000, ptr %2137, align 4
  %2138 = add i64 %2134, 1
  br label %2133

2139:                                             ; preds = %2133
  %2140 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), i64 64))
  %2141 = ptrtoint ptr %2140 to i64
  %2142 = add i64 %2141, 63
  %2143 = urem i64 %2142, 64
  %2144 = sub i64 %2142, %2143
  %2145 = inttoptr i64 %2144 to ptr
  br label %2146

2146:                                             ; preds = %2149, %2139
  %2147 = phi i64 [ %2151, %2149 ], [ 0, %2139 ]
  %2148 = icmp slt i64 %2147, 1
  br i1 %2148, label %2149, label %2152

2149:                                             ; preds = %2146
  %2150 = getelementptr i64, ptr %2145, i64 %2147
  store i64 0, ptr %2150, align 4
  %2151 = add i64 %2147, 1
  br label %2146

2152:                                             ; preds = %2146
  br label %2153

2153:                                             ; preds = %2189, %2152
  %2154 = phi i64 [ %2190, %2189 ], [ 0, %2152 ]
  %2155 = icmp slt i64 %2154, 32000
  br i1 %2155, label %2156, label %2191

2156:                                             ; preds = %2153
  br label %2157

2157:                                             ; preds = %2187, %2156
  %2158 = phi i64 [ %2188, %2187 ], [ 0, %2156 ]
  %2159 = icmp slt i64 %2158, 128
  br i1 %2159, label %2160, label %2189

2160:                                             ; preds = %2157
  %2161 = add i64 %2154, %2158
  br label %2162

2162:                                             ; preds = %2185, %2160
  %2163 = phi i64 [ %2186, %2185 ], [ 0, %2160 ]
  %2164 = icmp slt i64 %2163, 1
  br i1 %2164, label %2165, label %2187

2165:                                             ; preds = %2162
  br label %2166

2166:                                             ; preds = %2169, %2165
  %2167 = phi i64 [ %2184, %2169 ], [ 0, %2165 ]
  %2168 = icmp slt i64 %2167, 32
  br i1 %2168, label %2169, label %2185

2169:                                             ; preds = %2166
  %2170 = getelementptr float, ptr %2037, i64 %2161
  %2171 = mul i64 %2163, 32000
  %2172 = add i64 %2171, %2167
  %2173 = getelementptr float, ptr %2170, i64 %2172
  %2174 = load float, ptr %2173, align 4
  %2175 = getelementptr float, ptr %2132, i64 %2163
  %2176 = load float, ptr %2175, align 4
  %2177 = getelementptr i64, ptr %2145, i64 %2163
  %2178 = load i64, ptr %2177, align 4
  %2179 = add i64 %2154, %2167
  %2180 = add i64 %2179, %2158
  %2181 = fcmp ogt float %2174, %2176
  %2182 = select i1 %2181, float %2174, float %2176
  %2183 = select i1 %2181, i64 %2180, i64 %2178
  store float %2182, ptr %2175, align 4
  store i64 %2183, ptr %2177, align 4
  %2184 = add i64 %2167, 1
  br label %2166

2185:                                             ; preds = %2166
  %2186 = add i64 %2163, 1
  br label %2162

2187:                                             ; preds = %2162
  %2188 = add i64 %2158, 32
  br label %2157

2189:                                             ; preds = %2157
  %2190 = add i64 %2154, 128
  br label %2153

2191:                                             ; preds = %2153
  %2192 = load i64, ptr %2145, align 4
  call void @decode(i64 %36, i64 %2192)
  br label %31

2193:                                             ; preds = %31
  call void @end(i64 128)
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
declare i64 @llvm.smin.i64(i64, i64) #1

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
