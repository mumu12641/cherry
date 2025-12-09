; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__constant_32x2048x128xf32 = private constant [32 x [2048 x [128 x float]]] zeroinitializer, align 64
@__constant_8x8xf32 = private constant [8 x [8 x float]] [[8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03], [8 x float] [float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03, float 3.214000e+03]], align 64

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

declare void @print_memref_f32(i64, ptr)

define void @host() {
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

16:                                               ; preds = %324, %0
  %17 = phi i64 [ %325, %324 ], [ 0, %0 ]
  %18 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %133, %324 ], [ %15, %0 ]
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
  br i1 %19, label %45, label %326

45:                                               ; preds = %16
  %46 = phi i64 [ %17, %16 ]
  %47 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %34, %16 ]
  %48 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32000) to i64), i64 64))
  %49 = ptrtoint ptr %48 to i64
  %50 = add i64 %49, 63
  %51 = urem i64 %50, 64
  %52 = sub i64 %50, %51
  %53 = inttoptr i64 %52 to ptr
  br label %54

54:                                               ; preds = %73, %45
  %55 = phi i64 [ %75, %73 ], [ 0, %45 ]
  %56 = icmp slt i64 %55, 32000
  br i1 %56, label %57, label %76

57:                                               ; preds = %54
  br label %58

58:                                               ; preds = %71, %57
  %59 = phi i64 [ %72, %71 ], [ 0, %57 ]
  %60 = icmp slt i64 %59, 1
  br i1 %60, label %61, label %73

61:                                               ; preds = %58
  br label %62

62:                                               ; preds = %65, %61
  %63 = phi i64 [ %70, %65 ], [ 0, %61 ]
  %64 = icmp slt i64 %63, 8
  br i1 %64, label %65, label %71

65:                                               ; preds = %62
  %66 = getelementptr float, ptr %53, i64 %55
  %67 = mul i64 %59, 32000
  %68 = add i64 %67, %63
  %69 = getelementptr float, ptr %66, i64 %68
  store float 0.000000e+00, ptr %69, align 4
  %70 = add i64 %63, 1
  br label %62

71:                                               ; preds = %62
  %72 = add i64 %59, 1
  br label %58

73:                                               ; preds = %58
  %74 = getelementptr float, ptr %53, i64 %55
  call void @llvm.memcpy.p0.p0.i64(ptr %74, ptr %74, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %75 = add i64 %55, 8
  br label %54

76:                                               ; preds = %54
  br label %77

77:                                               ; preds = %116, %76
  %78 = phi i64 [ %117, %116 ], [ 0, %76 ]
  %79 = icmp slt i64 %78, 32000
  br i1 %79, label %80, label %118

80:                                               ; preds = %77
  br label %81

81:                                               ; preds = %113, %80
  %82 = phi i64 [ %115, %113 ], [ 0, %80 ]
  %83 = icmp slt i64 %82, 768
  br i1 %83, label %84, label %116

84:                                               ; preds = %81
  br label %85

85:                                               ; preds = %111, %84
  %86 = phi i64 [ %112, %111 ], [ 0, %84 ]
  %87 = icmp slt i64 %86, 1
  br i1 %87, label %88, label %113

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %109, %88
  %90 = phi i64 [ %110, %109 ], [ 0, %88 ]
  %91 = icmp slt i64 %90, 8
  br i1 %91, label %92, label %111

92:                                               ; preds = %89
  br label %93

93:                                               ; preds = %96, %92
  %94 = phi i64 [ %108, %96 ], [ 0, %92 ]
  %95 = icmp slt i64 %94, 8
  br i1 %95, label %96, label %109

96:                                               ; preds = %93
  %97 = mul i64 %94, 8
  %98 = add i64 %97, %90
  %99 = getelementptr float, ptr @__constant_8x8xf32, i64 %98
  %100 = load float, ptr %99, align 4
  %101 = getelementptr float, ptr %53, i64 %78
  %102 = mul i64 %86, 32000
  %103 = add i64 %102, %90
  %104 = getelementptr float, ptr %101, i64 %103
  %105 = load float, ptr %104, align 4
  %106 = fmul float %100, 5.000000e-01
  %107 = fadd float %105, %106
  store float %107, ptr %104, align 4
  %108 = add i64 %94, 1
  br label %93

109:                                              ; preds = %93
  %110 = add i64 %90, 1
  br label %89

111:                                              ; preds = %89
  %112 = add i64 %86, 1
  br label %85

113:                                              ; preds = %85
  %114 = getelementptr float, ptr %53, i64 %78
  call void @llvm.memcpy.p0.p0.i64(ptr %114, ptr %114, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %115 = add i64 %82, 8
  br label %81

116:                                              ; preds = %81
  %117 = add i64 %78, 8
  br label %77

118:                                              ; preds = %77
  %119 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8388608) to i64), i64 64))
  %120 = ptrtoint ptr %119 to i64
  %121 = add i64 %120, 63
  %122 = urem i64 %121, 64
  %123 = sub i64 %121, %122
  %124 = inttoptr i64 %123 to ptr
  %125 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %119, 0
  %126 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %125, ptr %124, 1
  %127 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, i64 0, 2
  %128 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %127, i64 32, 3, 0
  %129 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %128, i64 2048, 3, 1
  %130 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %129, i64 128, 3, 2
  %131 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %130, i64 262144, 4, 0
  %132 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %131, i64 128, 4, 1
  %133 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %132, i64 1, 4, 2
  br label %134

134:                                              ; preds = %191, %118
  %135 = phi i64 [ %192, %191 ], [ 0, %118 ]
  %136 = icmp slt i64 %135, 32
  br i1 %136, label %137, label %193

137:                                              ; preds = %134
  br label %138

138:                                              ; preds = %189, %137
  %139 = phi i64 [ %190, %189 ], [ 0, %137 ]
  %140 = icmp slt i64 %139, 2048
  br i1 %140, label %141, label %191

141:                                              ; preds = %138
  br label %142

142:                                              ; preds = %180, %141
  %143 = phi i64 [ %188, %180 ], [ 0, %141 ]
  %144 = icmp slt i64 %143, 128
  br i1 %144, label %145, label %189

145:                                              ; preds = %142
  %146 = mul i64 %135, 262144
  %147 = mul i64 %139, 128
  %148 = add i64 %146, %147
  %149 = add i64 %148, %143
  %150 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, i64 %149, 2
  %151 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %150, i64 8, 3, 0
  %152 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %151, i64 262144, 4, 0
  %153 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %152, i64 8, 3, 1
  %154 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %153, i64 128, 4, 1
  %155 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %154, i64 8, 3, 2
  %156 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %155, i64 1, 4, 2
  br label %157

157:                                              ; preds = %178, %145
  %158 = phi i64 [ %179, %178 ], [ 0, %145 ]
  %159 = icmp slt i64 %158, 8
  br i1 %159, label %160, label %180

160:                                              ; preds = %157
  br label %161

161:                                              ; preds = %176, %160
  %162 = phi i64 [ %177, %176 ], [ 0, %160 ]
  %163 = icmp slt i64 %162, 8
  br i1 %163, label %164, label %178

164:                                              ; preds = %161
  br label %165

165:                                              ; preds = %168, %164
  %166 = phi i64 [ %175, %168 ], [ 0, %164 ]
  %167 = icmp slt i64 %166, 8
  br i1 %167, label %168, label %176

168:                                              ; preds = %165
  %169 = getelementptr float, ptr %124, i64 %149
  %170 = mul i64 %158, 262144
  %171 = mul i64 %162, 128
  %172 = add i64 %170, %171
  %173 = add i64 %172, %166
  %174 = getelementptr float, ptr %169, i64 %173
  store float 0.000000e+00, ptr %174, align 4
  %175 = add i64 %166, 1
  br label %165

176:                                              ; preds = %165
  %177 = add i64 %162, 1
  br label %161

178:                                              ; preds = %161
  %179 = add i64 %158, 1
  br label %157

180:                                              ; preds = %157
  %181 = call ptr @llvm.stacksave.p0()
  %182 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %156, ptr %182, align 8
  %183 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %182, 1
  %184 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %156, ptr %184, align 8
  %185 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %184, 1
  %186 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %183, ptr %186, align 8
  %187 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %185, ptr %187, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %186, ptr %187)
  call void @llvm.stackrestore.p0(ptr %181)
  %188 = add i64 %143, 8
  br label %142

189:                                              ; preds = %142
  %190 = add i64 %139, 8
  br label %138

191:                                              ; preds = %138
  %192 = add i64 %135, 8
  br label %134

193:                                              ; preds = %134
  br label %194

194:                                              ; preds = %265, %193
  %195 = phi i64 [ %266, %265 ], [ 0, %193 ]
  %196 = icmp slt i64 %195, 32
  br i1 %196, label %197, label %267

197:                                              ; preds = %194
  br label %198

198:                                              ; preds = %263, %197
  %199 = phi i64 [ %264, %263 ], [ 0, %197 ]
  %200 = icmp slt i64 %199, 2048
  br i1 %200, label %201, label %265

201:                                              ; preds = %198
  br label %202

202:                                              ; preds = %254, %201
  %203 = phi i64 [ %262, %254 ], [ 0, %201 ]
  %204 = icmp slt i64 %203, 128
  br i1 %204, label %205, label %263

205:                                              ; preds = %202
  %206 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %47, 1
  %207 = mul i64 %195, 262144
  %208 = mul i64 %199, 128
  %209 = add i64 %207, %208
  %210 = add i64 %209, %203
  %211 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, i64 %210, 2
  %212 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %211, i64 8, 3, 0
  %213 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %212, i64 262144, 4, 0
  %214 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %213, i64 8, 3, 1
  %215 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %214, i64 128, 4, 1
  %216 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %215, i64 8, 3, 2
  %217 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %216, i64 1, 4, 2
  br label %218

218:                                              ; preds = %252, %205
  %219 = phi i64 [ %253, %252 ], [ 0, %205 ]
  %220 = icmp slt i64 %219, 8
  br i1 %220, label %221, label %254

221:                                              ; preds = %218
  br label %222

222:                                              ; preds = %250, %221
  %223 = phi i64 [ %251, %250 ], [ 0, %221 ]
  %224 = icmp slt i64 %223, 8
  br i1 %224, label %225, label %252

225:                                              ; preds = %222
  br label %226

226:                                              ; preds = %248, %225
  %227 = phi i64 [ %249, %248 ], [ 0, %225 ]
  %228 = icmp slt i64 %227, 8
  br i1 %228, label %229, label %250

229:                                              ; preds = %226
  br label %230

230:                                              ; preds = %233, %229
  %231 = phi i64 [ %247, %233 ], [ 0, %229 ]
  %232 = icmp slt i64 %231, 128
  br i1 %232, label %233, label %248

233:                                              ; preds = %230
  %234 = getelementptr float, ptr %206, i64 %209
  %235 = mul i64 %219, 262144
  %236 = mul i64 %223, 128
  %237 = add i64 %235, %236
  %238 = add i64 %237, %231
  %239 = getelementptr float, ptr %234, i64 %238
  %240 = load float, ptr %239, align 4
  %241 = getelementptr float, ptr %124, i64 %210
  %242 = add i64 %237, %227
  %243 = getelementptr float, ptr %241, i64 %242
  %244 = load float, ptr %243, align 4
  %245 = fmul float %240, 1.234200e+04
  %246 = fadd float %244, %245
  store float %246, ptr %243, align 4
  %247 = add i64 %231, 1
  br label %230

248:                                              ; preds = %230
  %249 = add i64 %227, 1
  br label %226

250:                                              ; preds = %226
  %251 = add i64 %223, 1
  br label %222

252:                                              ; preds = %222
  %253 = add i64 %219, 1
  br label %218

254:                                              ; preds = %218
  %255 = call ptr @llvm.stacksave.p0()
  %256 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %217, ptr %256, align 8
  %257 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %256, 1
  %258 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %217, ptr %258, align 8
  %259 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %258, 1
  %260 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %257, ptr %260, align 8
  %261 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %259, ptr %261, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %260, ptr %261)
  call void @llvm.stackrestore.p0(ptr %255)
  %262 = add i64 %203, 8
  br label %202

263:                                              ; preds = %202
  %264 = add i64 %199, 8
  br label %198

265:                                              ; preds = %198
  %266 = add i64 %195, 8
  br label %194

267:                                              ; preds = %194
  %268 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %269 = ptrtoint ptr %268 to i64
  %270 = add i64 %269, 63
  %271 = urem i64 %270, 64
  %272 = sub i64 %270, %271
  %273 = inttoptr i64 %272 to ptr
  br label %274

274:                                              ; preds = %277, %267
  %275 = phi i64 [ %279, %277 ], [ 0, %267 ]
  %276 = icmp slt i64 %275, 1
  br i1 %276, label %277, label %280

277:                                              ; preds = %274
  %278 = getelementptr float, ptr %273, i64 %275
  store float 0xFFF0000000000000, ptr %278, align 4
  %279 = add i64 %275, 1
  br label %274

280:                                              ; preds = %274
  %281 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), i64 64))
  %282 = ptrtoint ptr %281 to i64
  %283 = add i64 %282, 63
  %284 = urem i64 %283, 64
  %285 = sub i64 %283, %284
  %286 = inttoptr i64 %285 to ptr
  br label %287

287:                                              ; preds = %290, %280
  %288 = phi i64 [ %292, %290 ], [ 0, %280 ]
  %289 = icmp slt i64 %288, 1
  br i1 %289, label %290, label %293

290:                                              ; preds = %287
  %291 = getelementptr i64, ptr %286, i64 %288
  store i64 0, ptr %291, align 4
  %292 = add i64 %288, 1
  br label %287

293:                                              ; preds = %287
  br label %294

294:                                              ; preds = %322, %293
  %295 = phi i64 [ %323, %322 ], [ 0, %293 ]
  %296 = icmp slt i64 %295, 32000
  br i1 %296, label %297, label %324

297:                                              ; preds = %294
  br label %298

298:                                              ; preds = %320, %297
  %299 = phi i64 [ %321, %320 ], [ 0, %297 ]
  %300 = icmp slt i64 %299, 1
  br i1 %300, label %301, label %322

301:                                              ; preds = %298
  br label %302

302:                                              ; preds = %305, %301
  %303 = phi i64 [ %319, %305 ], [ 0, %301 ]
  %304 = icmp slt i64 %303, 8
  br i1 %304, label %305, label %320

305:                                              ; preds = %302
  %306 = getelementptr float, ptr %53, i64 %295
  %307 = mul i64 %299, 32000
  %308 = add i64 %307, %303
  %309 = getelementptr float, ptr %306, i64 %308
  %310 = load float, ptr %309, align 4
  %311 = getelementptr float, ptr %273, i64 %299
  %312 = load float, ptr %311, align 4
  %313 = getelementptr i64, ptr %286, i64 %299
  %314 = load i64, ptr %313, align 4
  %315 = add i64 %303, %295
  %316 = fcmp ogt float %310, %312
  %317 = select i1 %316, float %310, float %312
  %318 = select i1 %316, i64 %315, i64 %314
  store float %317, ptr %311, align 4
  store i64 %318, ptr %313, align 4
  %319 = add i64 %303, 1
  br label %302

320:                                              ; preds = %302
  %321 = add i64 %299, 1
  br label %298

322:                                              ; preds = %298
  %323 = add i64 %295, 8
  br label %294

324:                                              ; preds = %294
  %325 = add i64 %46, 1
  br label %16

326:                                              ; preds = %16
  %327 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, ptr %327, align 8
  call void @print_memref_f32(i64 3, ptr %327)
  ret void
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
