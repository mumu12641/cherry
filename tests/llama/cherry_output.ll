; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__constant_32x2048x128xf32 = private constant [32 x [2048 x [128 x float]]] zeroinitializer, align 64
@__constant_8x8xf32 = private constant [8 x [8 x float]] [[8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03]], align 64

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

declare void @print_memref_f32(i64, ptr)

define { ptr, ptr, i64, [3 x i64], [3 x i64] } @host() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8388608) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1, 0
  %8 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %8, i64 0, 2
  %10 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %9, i64 32, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, i64 2048, 3, 1
  %12 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %11, i64 128, 3, 2
  %13 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %12, i64 262144, 4, 0
  %14 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, i64 128, 4, 1
  %15 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @__constant_32x2048x128xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8388608), i1 false)
  br label %16

16:                                               ; preds = %332, %0
  %17 = phi i64 [ %333, %332 ], [ 0, %0 ]
  %18 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %140, %332 ], [ %15, %0 ]
  %19 = icmp slt i64 %17, 10
  %20 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8388608) to i64), i64 64))
  %21 = ptrtoint ptr %20 to i64
  %22 = add i64 %21, 63
  %23 = urem i64 %22, 64
  %24 = sub i64 %22, %23
  %25 = inttoptr i64 %24 to ptr
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %20, 0
  %27 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %26, ptr %25, 1
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, i64 0, 2
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, i64 32, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, i64 2048, 3, 1
  %31 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %30, i64 128, 3, 2
  %32 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, i64 262144, 4, 0
  %33 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %32, i64 128, 4, 1
  %34 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, i64 1, 4, 2
  %35 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %18, 3, 0
  %36 = mul i64 %35, 1
  %37 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %18, 3, 1
  %38 = mul i64 %36, %37
  %39 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %18, 3, 2
  %40 = mul i64 %38, %39
  %41 = mul i64 %40, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %42 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %18, 1
  %43 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %18, 2
  %44 = getelementptr float, ptr %42, i64 %43
  call void @llvm.memcpy.p0.p0.i64(ptr %25, ptr %44, i64 %41, i1 false)
  br i1 %19, label %45, label %334

45:                                               ; preds = %16
  %46 = phi i64 [ %17, %16 ]
  %47 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %34, %16 ]
  %48 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32000) to i64), i64 64))
  %49 = ptrtoint ptr %48 to i64
  %50 = add i64 %49, 63
  %51 = urem i64 %50, 64
  %52 = sub i64 %50, %51
  %53 = inttoptr i64 %52 to ptr
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %48, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, ptr %53, 1
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 0, 2
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 1, 3, 0
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, i64 32000, 3, 1
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, i64 32000, 4, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, i64 1, 4, 1
  br label %61

61:                                               ; preds = %80, %45
  %62 = phi i64 [ %82, %80 ], [ 0, %45 ]
  %63 = icmp slt i64 %62, 32000
  br i1 %63, label %64, label %83

64:                                               ; preds = %61
  br label %65

65:                                               ; preds = %78, %64
  %66 = phi i64 [ %79, %78 ], [ 0, %64 ]
  %67 = icmp slt i64 %66, 1
  br i1 %67, label %68, label %80

68:                                               ; preds = %65
  br label %69

69:                                               ; preds = %72, %68
  %70 = phi i64 [ %77, %72 ], [ 0, %68 ]
  %71 = icmp slt i64 %70, 8
  br i1 %71, label %72, label %78

72:                                               ; preds = %69
  %73 = getelementptr float, ptr %53, i64 %62
  %74 = mul i64 %66, 32000
  %75 = add i64 %74, %70
  %76 = getelementptr float, ptr %73, i64 %75
  store float 0.000000e+00, ptr %76, align 4
  %77 = add i64 %70, 1
  br label %69

78:                                               ; preds = %69
  %79 = add i64 %66, 1
  br label %65

80:                                               ; preds = %65
  %81 = getelementptr float, ptr %53, i64 %62
  call void @llvm.memcpy.p0.p0.i64(ptr %81, ptr %81, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %82 = add i64 %62, 8
  br label %61

83:                                               ; preds = %61
  br label %84

84:                                               ; preds = %123, %83
  %85 = phi i64 [ %124, %123 ], [ 0, %83 ]
  %86 = icmp slt i64 %85, 32000
  br i1 %86, label %87, label %125

87:                                               ; preds = %84
  br label %88

88:                                               ; preds = %120, %87
  %89 = phi i64 [ %122, %120 ], [ 0, %87 ]
  %90 = icmp slt i64 %89, 768
  br i1 %90, label %91, label %123

91:                                               ; preds = %88
  br label %92

92:                                               ; preds = %118, %91
  %93 = phi i64 [ %119, %118 ], [ 0, %91 ]
  %94 = icmp slt i64 %93, 1
  br i1 %94, label %95, label %120

95:                                               ; preds = %92
  br label %96

96:                                               ; preds = %116, %95
  %97 = phi i64 [ %117, %116 ], [ 0, %95 ]
  %98 = icmp slt i64 %97, 8
  br i1 %98, label %99, label %118

99:                                               ; preds = %96
  br label %100

100:                                              ; preds = %103, %99
  %101 = phi i64 [ %115, %103 ], [ 0, %99 ]
  %102 = icmp slt i64 %101, 8
  br i1 %102, label %103, label %116

103:                                              ; preds = %100
  %104 = mul i64 %101, 8
  %105 = add i64 %104, %97
  %106 = getelementptr float, ptr @__constant_8x8xf32, i64 %105
  %107 = load float, ptr %106, align 4
  %108 = getelementptr float, ptr %53, i64 %85
  %109 = mul i64 %93, 32000
  %110 = add i64 %109, %97
  %111 = getelementptr float, ptr %108, i64 %110
  %112 = load float, ptr %111, align 4
  %113 = fmul float %107, 5.000000e-01
  %114 = fadd float %112, %113
  store float %114, ptr %111, align 4
  %115 = add i64 %101, 1
  br label %100

116:                                              ; preds = %100
  %117 = add i64 %97, 1
  br label %96

118:                                              ; preds = %96
  %119 = add i64 %93, 1
  br label %92

120:                                              ; preds = %92
  %121 = getelementptr float, ptr %53, i64 %85
  call void @llvm.memcpy.p0.p0.i64(ptr %121, ptr %121, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %122 = add i64 %89, 8
  br label %88

123:                                              ; preds = %88
  %124 = add i64 %85, 8
  br label %84

125:                                              ; preds = %84
  %126 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8388608) to i64), i64 64))
  %127 = ptrtoint ptr %126 to i64
  %128 = add i64 %127, 63
  %129 = urem i64 %128, 64
  %130 = sub i64 %128, %129
  %131 = inttoptr i64 %130 to ptr
  %132 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %126, 0
  %133 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %132, ptr %131, 1
  %134 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %133, i64 0, 2
  %135 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %134, i64 32, 3, 0
  %136 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %135, i64 2048, 3, 1
  %137 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %136, i64 128, 3, 2
  %138 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %137, i64 262144, 4, 0
  %139 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %138, i64 128, 4, 1
  %140 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %139, i64 1, 4, 2
  br label %141

141:                                              ; preds = %198, %125
  %142 = phi i64 [ %199, %198 ], [ 0, %125 ]
  %143 = icmp slt i64 %142, 32
  br i1 %143, label %144, label %200

144:                                              ; preds = %141
  br label %145

145:                                              ; preds = %196, %144
  %146 = phi i64 [ %197, %196 ], [ 0, %144 ]
  %147 = icmp slt i64 %146, 2048
  br i1 %147, label %148, label %198

148:                                              ; preds = %145
  br label %149

149:                                              ; preds = %187, %148
  %150 = phi i64 [ %195, %187 ], [ 0, %148 ]
  %151 = icmp slt i64 %150, 128
  br i1 %151, label %152, label %196

152:                                              ; preds = %149
  %153 = mul i64 %142, 262144
  %154 = mul i64 %146, 128
  %155 = add i64 %153, %154
  %156 = add i64 %155, %150
  %157 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %133, i64 %156, 2
  %158 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %157, i64 8, 3, 0
  %159 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %158, i64 262144, 4, 0
  %160 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %159, i64 8, 3, 1
  %161 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %160, i64 128, 4, 1
  %162 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %161, i64 8, 3, 2
  %163 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %162, i64 1, 4, 2
  br label %164

164:                                              ; preds = %185, %152
  %165 = phi i64 [ %186, %185 ], [ 0, %152 ]
  %166 = icmp slt i64 %165, 8
  br i1 %166, label %167, label %187

167:                                              ; preds = %164
  br label %168

168:                                              ; preds = %183, %167
  %169 = phi i64 [ %184, %183 ], [ 0, %167 ]
  %170 = icmp slt i64 %169, 8
  br i1 %170, label %171, label %185

171:                                              ; preds = %168
  br label %172

172:                                              ; preds = %175, %171
  %173 = phi i64 [ %182, %175 ], [ 0, %171 ]
  %174 = icmp slt i64 %173, 8
  br i1 %174, label %175, label %183

175:                                              ; preds = %172
  %176 = getelementptr float, ptr %131, i64 %156
  %177 = mul i64 %165, 262144
  %178 = mul i64 %169, 128
  %179 = add i64 %177, %178
  %180 = add i64 %179, %173
  %181 = getelementptr float, ptr %176, i64 %180
  store float 0.000000e+00, ptr %181, align 4
  %182 = add i64 %173, 1
  br label %172

183:                                              ; preds = %172
  %184 = add i64 %169, 1
  br label %168

185:                                              ; preds = %168
  %186 = add i64 %165, 1
  br label %164

187:                                              ; preds = %164
  %188 = call ptr @llvm.stacksave.p0()
  %189 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %163, ptr %189, align 8
  %190 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %189, 1
  %191 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %163, ptr %191, align 8
  %192 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %191, 1
  %193 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %190, ptr %193, align 8
  %194 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %192, ptr %194, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %193, ptr %194)
  call void @llvm.stackrestore.p0(ptr %188)
  %195 = add i64 %150, 8
  br label %149

196:                                              ; preds = %149
  %197 = add i64 %146, 8
  br label %145

198:                                              ; preds = %145
  %199 = add i64 %142, 8
  br label %141

200:                                              ; preds = %141
  br label %201

201:                                              ; preds = %272, %200
  %202 = phi i64 [ %273, %272 ], [ 0, %200 ]
  %203 = icmp slt i64 %202, 32
  br i1 %203, label %204, label %274

204:                                              ; preds = %201
  br label %205

205:                                              ; preds = %270, %204
  %206 = phi i64 [ %271, %270 ], [ 0, %204 ]
  %207 = icmp slt i64 %206, 2048
  br i1 %207, label %208, label %272

208:                                              ; preds = %205
  br label %209

209:                                              ; preds = %261, %208
  %210 = phi i64 [ %269, %261 ], [ 0, %208 ]
  %211 = icmp slt i64 %210, 128
  br i1 %211, label %212, label %270

212:                                              ; preds = %209
  %213 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %47, 1
  %214 = mul i64 %202, 262144
  %215 = mul i64 %206, 128
  %216 = add i64 %214, %215
  %217 = add i64 %216, %210
  %218 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %133, i64 %217, 2
  %219 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %218, i64 8, 3, 0
  %220 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %219, i64 262144, 4, 0
  %221 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %220, i64 8, 3, 1
  %222 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %221, i64 128, 4, 1
  %223 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %222, i64 8, 3, 2
  %224 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %223, i64 1, 4, 2
  br label %225

225:                                              ; preds = %259, %212
  %226 = phi i64 [ %260, %259 ], [ 0, %212 ]
  %227 = icmp slt i64 %226, 8
  br i1 %227, label %228, label %261

228:                                              ; preds = %225
  br label %229

229:                                              ; preds = %257, %228
  %230 = phi i64 [ %258, %257 ], [ 0, %228 ]
  %231 = icmp slt i64 %230, 8
  br i1 %231, label %232, label %259

232:                                              ; preds = %229
  br label %233

233:                                              ; preds = %255, %232
  %234 = phi i64 [ %256, %255 ], [ 0, %232 ]
  %235 = icmp slt i64 %234, 8
  br i1 %235, label %236, label %257

236:                                              ; preds = %233
  br label %237

237:                                              ; preds = %240, %236
  %238 = phi i64 [ %254, %240 ], [ 0, %236 ]
  %239 = icmp slt i64 %238, 128
  br i1 %239, label %240, label %255

240:                                              ; preds = %237
  %241 = getelementptr float, ptr %213, i64 %216
  %242 = mul i64 %226, 262144
  %243 = mul i64 %230, 128
  %244 = add i64 %242, %243
  %245 = add i64 %244, %238
  %246 = getelementptr float, ptr %241, i64 %245
  %247 = load float, ptr %246, align 4
  %248 = getelementptr float, ptr %131, i64 %217
  %249 = add i64 %244, %234
  %250 = getelementptr float, ptr %248, i64 %249
  %251 = load float, ptr %250, align 4
  %252 = fmul float %247, 1.234200e+04
  %253 = fadd float %251, %252
  store float %253, ptr %250, align 4
  %254 = add i64 %238, 1
  br label %237

255:                                              ; preds = %237
  %256 = add i64 %234, 1
  br label %233

257:                                              ; preds = %233
  %258 = add i64 %230, 1
  br label %229

259:                                              ; preds = %229
  %260 = add i64 %226, 1
  br label %225

261:                                              ; preds = %225
  %262 = call ptr @llvm.stacksave.p0()
  %263 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %224, ptr %263, align 8
  %264 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %263, 1
  %265 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %224, ptr %265, align 8
  %266 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %265, 1
  %267 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %264, ptr %267, align 8
  %268 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %266, ptr %268, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %267, ptr %268)
  call void @llvm.stackrestore.p0(ptr %262)
  %269 = add i64 %210, 8
  br label %209

270:                                              ; preds = %209
  %271 = add i64 %206, 8
  br label %205

272:                                              ; preds = %205
  %273 = add i64 %202, 8
  br label %201

274:                                              ; preds = %201
  %275 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, ptr %275, align 8
  call void @print_memref_f32(i64 2, ptr %275)
  %276 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %277 = ptrtoint ptr %276 to i64
  %278 = add i64 %277, 63
  %279 = urem i64 %278, 64
  %280 = sub i64 %278, %279
  %281 = inttoptr i64 %280 to ptr
  br label %282

282:                                              ; preds = %285, %274
  %283 = phi i64 [ %287, %285 ], [ 0, %274 ]
  %284 = icmp slt i64 %283, 1
  br i1 %284, label %285, label %288

285:                                              ; preds = %282
  %286 = getelementptr float, ptr %281, i64 %283
  store float 0xFFF0000000000000, ptr %286, align 4
  %287 = add i64 %283, 1
  br label %282

288:                                              ; preds = %282
  %289 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), i64 64))
  %290 = ptrtoint ptr %289 to i64
  %291 = add i64 %290, 63
  %292 = urem i64 %291, 64
  %293 = sub i64 %291, %292
  %294 = inttoptr i64 %293 to ptr
  br label %295

295:                                              ; preds = %298, %288
  %296 = phi i64 [ %300, %298 ], [ 0, %288 ]
  %297 = icmp slt i64 %296, 1
  br i1 %297, label %298, label %301

298:                                              ; preds = %295
  %299 = getelementptr i64, ptr %294, i64 %296
  store i64 0, ptr %299, align 4
  %300 = add i64 %296, 1
  br label %295

301:                                              ; preds = %295
  br label %302

302:                                              ; preds = %330, %301
  %303 = phi i64 [ %331, %330 ], [ 0, %301 ]
  %304 = icmp slt i64 %303, 32000
  br i1 %304, label %305, label %332

305:                                              ; preds = %302
  br label %306

306:                                              ; preds = %328, %305
  %307 = phi i64 [ %329, %328 ], [ 0, %305 ]
  %308 = icmp slt i64 %307, 1
  br i1 %308, label %309, label %330

309:                                              ; preds = %306
  br label %310

310:                                              ; preds = %313, %309
  %311 = phi i64 [ %327, %313 ], [ 0, %309 ]
  %312 = icmp slt i64 %311, 8
  br i1 %312, label %313, label %328

313:                                              ; preds = %310
  %314 = getelementptr float, ptr %53, i64 %303
  %315 = mul i64 %307, 32000
  %316 = add i64 %315, %311
  %317 = getelementptr float, ptr %314, i64 %316
  %318 = load float, ptr %317, align 4
  %319 = getelementptr float, ptr %281, i64 %307
  %320 = load float, ptr %319, align 4
  %321 = getelementptr i64, ptr %294, i64 %307
  %322 = load i64, ptr %321, align 4
  %323 = add i64 %311, %303
  %324 = fcmp ogt float %318, %320
  %325 = select i1 %324, float %318, float %320
  %326 = select i1 %324, i64 %323, i64 %322
  store float %325, ptr %319, align 4
  store i64 %326, ptr %321, align 4
  %327 = add i64 %311, 1
  br label %310

328:                                              ; preds = %310
  %329 = add i64 %307, 1
  br label %306

330:                                              ; preds = %306
  %331 = add i64 %303, 8
  br label %302

332:                                              ; preds = %302
  %333 = add i64 %46, 1
  br label %16

334:                                              ; preds = %16
  ret { ptr, ptr, i64, [3 x i64], [3 x i64] } %34
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
