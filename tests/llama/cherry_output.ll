; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__constant_12x1024x768xf32 = private constant [12 x [1024 x [768 x float]]] zeroinitializer, align 64
@__constant_1x768xf32 = private constant [1 x [768 x float]] [[768 x float] [float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00]], align 64
@__constant_3xi64 = private constant [3 x i64] [i64 1, i64 1, i64 768], align 64

declare ptr @malloc(i64)

declare void @printMemrefI64(i64, ptr)

define void @host() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @__constant_12x1024x768xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9437184), i1 false)
  %7 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %8 = ptrtoint ptr %7 to i64
  %9 = add i64 %8, 63
  %10 = urem i64 %9, 64
  %11 = sub i64 %9, %10
  %12 = inttoptr i64 %11 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %12, ptr @__constant_12x1024x768xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9437184), i1 false)
  br label %13

13:                                               ; preds = %491, %0
  %14 = phi i64 [ %493, %491 ], [ 0, %0 ]
  %15 = icmp slt i64 %14, 10
  br i1 %15, label %16, label %494

16:                                               ; preds = %13
  %17 = phi i64 [ %14, %13 ]
  %18 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %19 = ptrtoint ptr %18 to i64
  %20 = add i64 %19, 63
  %21 = urem i64 %20, 64
  %22 = sub i64 %20, %21
  %23 = inttoptr i64 %22 to ptr
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %18, 0
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, ptr %23, 1
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 0, 2
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 1, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 768, 3, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 768, 4, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 1, 4, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %23, ptr @__constant_1x768xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %31

31:                                               ; preds = %349, %16
  %32 = phi i64 [ %355, %349 ], [ 0, %16 ]
  %33 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %130, %349 ], [ %30, %16 ]
  %34 = icmp slt i64 %32, 12
  br i1 %34, label %35, label %356

35:                                               ; preds = %31
  %36 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %37 = ptrtoint ptr %36 to i64
  %38 = add i64 %37, 63
  %39 = urem i64 %38, 64
  %40 = sub i64 %38, %39
  %41 = inttoptr i64 %40 to ptr
  br label %42

42:                                               ; preds = %45, %35
  %43 = phi i64 [ %47, %45 ], [ 0, %35 ]
  %44 = icmp slt i64 %43, 1
  br i1 %44, label %45, label %48

45:                                               ; preds = %42
  %46 = getelementptr float, ptr %41, i64 %43
  store float 0.000000e+00, ptr %46, align 4
  %47 = add i64 %43, 1
  br label %42

48:                                               ; preds = %42
  br label %49

49:                                               ; preds = %74, %48
  %50 = phi i64 [ %75, %74 ], [ 0, %48 ]
  %51 = icmp slt i64 %50, 768
  br i1 %51, label %52, label %76

52:                                               ; preds = %49
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 1
  br label %54

54:                                               ; preds = %72, %52
  %55 = phi i64 [ %73, %72 ], [ 0, %52 ]
  %56 = icmp slt i64 %55, 1
  br i1 %56, label %57, label %74

57:                                               ; preds = %54
  br label %58

58:                                               ; preds = %61, %57
  %59 = phi i64 [ %71, %61 ], [ 0, %57 ]
  %60 = icmp slt i64 %59, 8
  br i1 %60, label %61, label %72

61:                                               ; preds = %58
  %62 = getelementptr float, ptr %53, i64 %50
  %63 = mul i64 %55, 768
  %64 = add i64 %63, %59
  %65 = getelementptr float, ptr %62, i64 %64
  %66 = load float, ptr %65, align 4
  %67 = getelementptr float, ptr %41, i64 %55
  %68 = load float, ptr %67, align 4
  %69 = fmul float %66, %66
  %70 = fadd float %68, %69
  store float %70, ptr %67, align 4
  %71 = add i64 %59, 1
  br label %58

72:                                               ; preds = %58
  %73 = add i64 %55, 1
  br label %54

74:                                               ; preds = %54
  %75 = add i64 %50, 8
  br label %49

76:                                               ; preds = %49
  %77 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %78 = ptrtoint ptr %77 to i64
  %79 = add i64 %78, 63
  %80 = urem i64 %79, 64
  %81 = sub i64 %79, %80
  %82 = inttoptr i64 %81 to ptr
  br label %83

83:                                               ; preds = %114, %76
  %84 = phi i64 [ %116, %114 ], [ 0, %76 ]
  %85 = icmp slt i64 %84, 768
  br i1 %85, label %86, label %117

86:                                               ; preds = %83
  %87 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 1
  br label %88

88:                                               ; preds = %112, %86
  %89 = phi i64 [ %113, %112 ], [ 0, %86 ]
  %90 = icmp slt i64 %89, 1
  br i1 %90, label %91, label %114

91:                                               ; preds = %88
  br label %92

92:                                               ; preds = %95, %91
  %93 = phi i64 [ %111, %95 ], [ 0, %91 ]
  %94 = icmp slt i64 %93, 8
  br i1 %94, label %95, label %112

95:                                               ; preds = %92
  %96 = getelementptr float, ptr %87, i64 %84
  %97 = mul i64 %89, 768
  %98 = add i64 %97, %93
  %99 = getelementptr float, ptr %96, i64 %98
  %100 = load float, ptr %99, align 4
  %101 = getelementptr float, ptr %41, i64 %89
  %102 = load float, ptr %101, align 4
  %103 = fdiv float %102, 7.680000e+02
  %104 = fadd float %103, 0x3EE4F8B580000000
  %105 = call float @llvm.sqrt.f32(float %104)
  %106 = fdiv float 1.000000e+00, %105
  %107 = fmul float %100, %106
  %108 = fmul float %107, 3.000000e+00
  %109 = getelementptr float, ptr %82, i64 %84
  %110 = getelementptr float, ptr %109, i64 %98
  store float %108, ptr %110, align 4
  %111 = add i64 %93, 1
  br label %92

112:                                              ; preds = %92
  %113 = add i64 %89, 1
  br label %88

114:                                              ; preds = %88
  %115 = getelementptr float, ptr %82, i64 %84
  call void @llvm.memcpy.p0.p0.i64(ptr %115, ptr %115, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %116 = add i64 %84, 8
  br label %83

117:                                              ; preds = %83
  %118 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %119 = ptrtoint ptr %118 to i64
  %120 = add i64 %119, 63
  %121 = urem i64 %120, 64
  %122 = sub i64 %120, %121
  %123 = inttoptr i64 %122 to ptr
  %124 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %118, 0
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %124, ptr %123, 1
  %126 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, i64 0, 2
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, i64 1, 3, 0
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, i64 768, 3, 1
  %129 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %128, i64 768, 4, 0
  %130 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %129, i64 1, 4, 1
  br label %131

131:                                              ; preds = %150, %117
  %132 = phi i64 [ %152, %150 ], [ 0, %117 ]
  %133 = icmp slt i64 %132, 768
  br i1 %133, label %134, label %153

134:                                              ; preds = %131
  br label %135

135:                                              ; preds = %148, %134
  %136 = phi i64 [ %149, %148 ], [ 0, %134 ]
  %137 = icmp slt i64 %136, 1
  br i1 %137, label %138, label %150

138:                                              ; preds = %135
  br label %139

139:                                              ; preds = %142, %138
  %140 = phi i64 [ %147, %142 ], [ 0, %138 ]
  %141 = icmp slt i64 %140, 8
  br i1 %141, label %142, label %148

142:                                              ; preds = %139
  %143 = getelementptr float, ptr %123, i64 %132
  %144 = mul i64 %136, 768
  %145 = add i64 %144, %140
  %146 = getelementptr float, ptr %143, i64 %145
  store float 0.000000e+00, ptr %146, align 4
  %147 = add i64 %140, 1
  br label %139

148:                                              ; preds = %139
  %149 = add i64 %136, 1
  br label %135

150:                                              ; preds = %135
  %151 = getelementptr float, ptr %123, i64 %132
  call void @llvm.memcpy.p0.p0.i64(ptr %151, ptr %151, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %152 = add i64 %132, 8
  br label %131

153:                                              ; preds = %131
  br label %154

154:                                              ; preds = %193, %153
  %155 = phi i64 [ %194, %193 ], [ 0, %153 ]
  %156 = icmp slt i64 %155, 768
  br i1 %156, label %157, label %195

157:                                              ; preds = %154
  br label %158

158:                                              ; preds = %190, %157
  %159 = phi i64 [ %192, %190 ], [ 0, %157 ]
  %160 = icmp slt i64 %159, 768
  br i1 %160, label %161, label %193

161:                                              ; preds = %158
  br label %162

162:                                              ; preds = %188, %161
  %163 = phi i64 [ %189, %188 ], [ 0, %161 ]
  %164 = icmp slt i64 %163, 1
  br i1 %164, label %165, label %190

165:                                              ; preds = %162
  br label %166

166:                                              ; preds = %186, %165
  %167 = phi i64 [ %187, %186 ], [ 0, %165 ]
  %168 = icmp slt i64 %167, 8
  br i1 %168, label %169, label %188

169:                                              ; preds = %166
  br label %170

170:                                              ; preds = %173, %169
  %171 = phi i64 [ %185, %173 ], [ 0, %169 ]
  %172 = icmp slt i64 %171, 8
  br i1 %172, label %173, label %186

173:                                              ; preds = %170
  %174 = getelementptr float, ptr %82, i64 %159
  %175 = mul i64 %163, 768
  %176 = add i64 %175, %171
  %177 = getelementptr float, ptr %174, i64 %176
  %178 = load float, ptr %177, align 4
  %179 = getelementptr float, ptr %123, i64 %155
  %180 = add i64 %175, %167
  %181 = getelementptr float, ptr %179, i64 %180
  %182 = load float, ptr %181, align 4
  %183 = fmul float %178, 4.000000e+00
  %184 = fadd float %182, %183
  store float %184, ptr %181, align 4
  %185 = add i64 %171, 1
  br label %170

186:                                              ; preds = %170
  %187 = add i64 %167, 1
  br label %166

188:                                              ; preds = %166
  %189 = add i64 %163, 1
  br label %162

190:                                              ; preds = %162
  %191 = getelementptr float, ptr %123, i64 %155
  call void @llvm.memcpy.p0.p0.i64(ptr %191, ptr %191, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %192 = add i64 %159, 8
  br label %158

193:                                              ; preds = %158
  %194 = add i64 %155, 8
  br label %154

195:                                              ; preds = %154
  %196 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %197 = ptrtoint ptr %196 to i64
  %198 = add i64 %197, 63
  %199 = urem i64 %198, 64
  %200 = sub i64 %198, %199
  %201 = inttoptr i64 %200 to ptr
  br label %202

202:                                              ; preds = %221, %195
  %203 = phi i64 [ %223, %221 ], [ 0, %195 ]
  %204 = icmp slt i64 %203, 768
  br i1 %204, label %205, label %224

205:                                              ; preds = %202
  br label %206

206:                                              ; preds = %219, %205
  %207 = phi i64 [ %220, %219 ], [ 0, %205 ]
  %208 = icmp slt i64 %207, 1
  br i1 %208, label %209, label %221

209:                                              ; preds = %206
  br label %210

210:                                              ; preds = %213, %209
  %211 = phi i64 [ %218, %213 ], [ 0, %209 ]
  %212 = icmp slt i64 %211, 8
  br i1 %212, label %213, label %219

213:                                              ; preds = %210
  %214 = getelementptr float, ptr %201, i64 %203
  %215 = mul i64 %207, 768
  %216 = add i64 %215, %211
  %217 = getelementptr float, ptr %214, i64 %216
  store float 0.000000e+00, ptr %217, align 4
  %218 = add i64 %211, 1
  br label %210

219:                                              ; preds = %210
  %220 = add i64 %207, 1
  br label %206

221:                                              ; preds = %206
  %222 = getelementptr float, ptr %201, i64 %203
  call void @llvm.memcpy.p0.p0.i64(ptr %222, ptr %222, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %223 = add i64 %203, 8
  br label %202

224:                                              ; preds = %202
  %225 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %226 = ptrtoint ptr %225 to i64
  %227 = add i64 %226, 63
  %228 = urem i64 %227, 64
  %229 = sub i64 %227, %228
  %230 = inttoptr i64 %229 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %230, ptr %201, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %231

231:                                              ; preds = %270, %224
  %232 = phi i64 [ %271, %270 ], [ 0, %224 ]
  %233 = icmp slt i64 %232, 768
  br i1 %233, label %234, label %272

234:                                              ; preds = %231
  br label %235

235:                                              ; preds = %267, %234
  %236 = phi i64 [ %269, %267 ], [ 0, %234 ]
  %237 = icmp slt i64 %236, 768
  br i1 %237, label %238, label %270

238:                                              ; preds = %235
  br label %239

239:                                              ; preds = %265, %238
  %240 = phi i64 [ %266, %265 ], [ 0, %238 ]
  %241 = icmp slt i64 %240, 1
  br i1 %241, label %242, label %267

242:                                              ; preds = %239
  br label %243

243:                                              ; preds = %263, %242
  %244 = phi i64 [ %264, %263 ], [ 0, %242 ]
  %245 = icmp slt i64 %244, 8
  br i1 %245, label %246, label %265

246:                                              ; preds = %243
  br label %247

247:                                              ; preds = %250, %246
  %248 = phi i64 [ %262, %250 ], [ 0, %246 ]
  %249 = icmp slt i64 %248, 8
  br i1 %249, label %250, label %263

250:                                              ; preds = %247
  %251 = getelementptr float, ptr %82, i64 %236
  %252 = mul i64 %240, 768
  %253 = add i64 %252, %248
  %254 = getelementptr float, ptr %251, i64 %253
  %255 = load float, ptr %254, align 4
  %256 = getelementptr float, ptr %230, i64 %232
  %257 = add i64 %252, %244
  %258 = getelementptr float, ptr %256, i64 %257
  %259 = load float, ptr %258, align 4
  %260 = fmul float %255, 5.000000e+00
  %261 = fadd float %259, %260
  store float %261, ptr %258, align 4
  %262 = add i64 %248, 1
  br label %247

263:                                              ; preds = %247
  %264 = add i64 %244, 1
  br label %243

265:                                              ; preds = %243
  %266 = add i64 %240, 1
  br label %239

267:                                              ; preds = %239
  %268 = getelementptr float, ptr %230, i64 %232
  call void @llvm.memcpy.p0.p0.i64(ptr %268, ptr %268, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %269 = add i64 %236, 8
  br label %235

270:                                              ; preds = %235
  %271 = add i64 %232, 8
  br label %231

272:                                              ; preds = %231
  %273 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %274 = ptrtoint ptr %273 to i64
  %275 = add i64 %274, 63
  %276 = urem i64 %275, 64
  %277 = sub i64 %275, %276
  %278 = inttoptr i64 %277 to ptr
  br label %279

279:                                              ; preds = %298, %272
  %280 = phi i64 [ %300, %298 ], [ 0, %272 ]
  %281 = icmp slt i64 %280, 768
  br i1 %281, label %282, label %301

282:                                              ; preds = %279
  br label %283

283:                                              ; preds = %296, %282
  %284 = phi i64 [ %297, %296 ], [ 0, %282 ]
  %285 = icmp slt i64 %284, 1
  br i1 %285, label %286, label %298

286:                                              ; preds = %283
  br label %287

287:                                              ; preds = %290, %286
  %288 = phi i64 [ %295, %290 ], [ 0, %286 ]
  %289 = icmp slt i64 %288, 8
  br i1 %289, label %290, label %296

290:                                              ; preds = %287
  %291 = getelementptr float, ptr %278, i64 %280
  %292 = mul i64 %284, 768
  %293 = add i64 %292, %288
  %294 = getelementptr float, ptr %291, i64 %293
  store float 0.000000e+00, ptr %294, align 4
  %295 = add i64 %288, 1
  br label %287

296:                                              ; preds = %287
  %297 = add i64 %284, 1
  br label %283

298:                                              ; preds = %283
  %299 = getelementptr float, ptr %278, i64 %280
  call void @llvm.memcpy.p0.p0.i64(ptr %299, ptr %299, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %300 = add i64 %280, 8
  br label %279

301:                                              ; preds = %279
  %302 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %303 = ptrtoint ptr %302 to i64
  %304 = add i64 %303, 63
  %305 = urem i64 %304, 64
  %306 = sub i64 %304, %305
  %307 = inttoptr i64 %306 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %307, ptr %278, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %308

308:                                              ; preds = %347, %301
  %309 = phi i64 [ %348, %347 ], [ 0, %301 ]
  %310 = icmp slt i64 %309, 768
  br i1 %310, label %311, label %349

311:                                              ; preds = %308
  br label %312

312:                                              ; preds = %344, %311
  %313 = phi i64 [ %346, %344 ], [ 0, %311 ]
  %314 = icmp slt i64 %313, 768
  br i1 %314, label %315, label %347

315:                                              ; preds = %312
  br label %316

316:                                              ; preds = %342, %315
  %317 = phi i64 [ %343, %342 ], [ 0, %315 ]
  %318 = icmp slt i64 %317, 1
  br i1 %318, label %319, label %344

319:                                              ; preds = %316
  br label %320

320:                                              ; preds = %340, %319
  %321 = phi i64 [ %341, %340 ], [ 0, %319 ]
  %322 = icmp slt i64 %321, 8
  br i1 %322, label %323, label %342

323:                                              ; preds = %320
  br label %324

324:                                              ; preds = %327, %323
  %325 = phi i64 [ %339, %327 ], [ 0, %323 ]
  %326 = icmp slt i64 %325, 8
  br i1 %326, label %327, label %340

327:                                              ; preds = %324
  %328 = getelementptr float, ptr %82, i64 %313
  %329 = mul i64 %317, 768
  %330 = add i64 %329, %325
  %331 = getelementptr float, ptr %328, i64 %330
  %332 = load float, ptr %331, align 4
  %333 = getelementptr float, ptr %307, i64 %309
  %334 = add i64 %329, %321
  %335 = getelementptr float, ptr %333, i64 %334
  %336 = load float, ptr %335, align 4
  %337 = fmul float %332, 6.000000e+00
  %338 = fadd float %336, %337
  store float %338, ptr %335, align 4
  %339 = add i64 %325, 1
  br label %324

340:                                              ; preds = %324
  %341 = add i64 %321, 1
  br label %320

342:                                              ; preds = %320
  %343 = add i64 %317, 1
  br label %316

344:                                              ; preds = %316
  %345 = getelementptr float, ptr %307, i64 %309
  call void @llvm.memcpy.p0.p0.i64(ptr %345, ptr %345, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %346 = add i64 %313, 8
  br label %312

347:                                              ; preds = %312
  %348 = add i64 %309, 8
  br label %308

349:                                              ; preds = %308
  %350 = mul i64 %32, 786432
  %351 = mul i64 %17, 768
  %352 = add i64 %350, %351
  %353 = getelementptr float, ptr %6, i64 %352
  call void @llvm.memcpy.p0.p0.i64(ptr %353, ptr %230, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %354 = getelementptr float, ptr %12, i64 %352
  call void @llvm.memcpy.p0.p0.i64(ptr %354, ptr %307, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %355 = add i64 %32, 1
  br label %31

356:                                              ; preds = %31
  %357 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32000) to i64), i64 64))
  %358 = ptrtoint ptr %357 to i64
  %359 = add i64 %358, 63
  %360 = urem i64 %359, 64
  %361 = sub i64 %359, %360
  %362 = inttoptr i64 %361 to ptr
  br label %363

363:                                              ; preds = %382, %356
  %364 = phi i64 [ %384, %382 ], [ 0, %356 ]
  %365 = icmp slt i64 %364, 32000
  br i1 %365, label %366, label %385

366:                                              ; preds = %363
  br label %367

367:                                              ; preds = %380, %366
  %368 = phi i64 [ %381, %380 ], [ 0, %366 ]
  %369 = icmp slt i64 %368, 1
  br i1 %369, label %370, label %382

370:                                              ; preds = %367
  br label %371

371:                                              ; preds = %374, %370
  %372 = phi i64 [ %379, %374 ], [ 0, %370 ]
  %373 = icmp slt i64 %372, 8
  br i1 %373, label %374, label %380

374:                                              ; preds = %371
  %375 = getelementptr float, ptr %362, i64 %364
  %376 = mul i64 %368, 32000
  %377 = add i64 %376, %372
  %378 = getelementptr float, ptr %375, i64 %377
  store float 0.000000e+00, ptr %378, align 4
  %379 = add i64 %372, 1
  br label %371

380:                                              ; preds = %371
  %381 = add i64 %368, 1
  br label %367

382:                                              ; preds = %367
  %383 = getelementptr float, ptr %362, i64 %364
  call void @llvm.memcpy.p0.p0.i64(ptr %383, ptr %383, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %384 = add i64 %364, 8
  br label %363

385:                                              ; preds = %363
  br label %386

386:                                              ; preds = %427, %385
  %387 = phi i64 [ %428, %427 ], [ 0, %385 ]
  %388 = icmp slt i64 %387, 32000
  br i1 %388, label %389, label %429

389:                                              ; preds = %386
  br label %390

390:                                              ; preds = %424, %389
  %391 = phi i64 [ %426, %424 ], [ 0, %389 ]
  %392 = icmp slt i64 %391, 768
  br i1 %392, label %393, label %427

393:                                              ; preds = %390
  %394 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 1
  br label %395

395:                                              ; preds = %422, %393
  %396 = phi i64 [ %423, %422 ], [ 0, %393 ]
  %397 = icmp slt i64 %396, 1
  br i1 %397, label %398, label %424

398:                                              ; preds = %395
  br label %399

399:                                              ; preds = %420, %398
  %400 = phi i64 [ %421, %420 ], [ 0, %398 ]
  %401 = icmp slt i64 %400, 8
  br i1 %401, label %402, label %422

402:                                              ; preds = %399
  br label %403

403:                                              ; preds = %406, %402
  %404 = phi i64 [ %419, %406 ], [ 0, %402 ]
  %405 = icmp slt i64 %404, 8
  br i1 %405, label %406, label %420

406:                                              ; preds = %403
  %407 = getelementptr float, ptr %394, i64 %391
  %408 = mul i64 %396, 768
  %409 = add i64 %408, %404
  %410 = getelementptr float, ptr %407, i64 %409
  %411 = load float, ptr %410, align 4
  %412 = getelementptr float, ptr %362, i64 %387
  %413 = mul i64 %396, 32000
  %414 = add i64 %413, %400
  %415 = getelementptr float, ptr %412, i64 %414
  %416 = load float, ptr %415, align 4
  %417 = fmul float %411, 1.300000e+01
  %418 = fadd float %416, %417
  store float %418, ptr %415, align 4
  %419 = add i64 %404, 1
  br label %403

420:                                              ; preds = %403
  %421 = add i64 %400, 1
  br label %399

422:                                              ; preds = %399
  %423 = add i64 %396, 1
  br label %395

424:                                              ; preds = %395
  %425 = getelementptr float, ptr %362, i64 %387
  call void @llvm.memcpy.p0.p0.i64(ptr %425, ptr %425, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %426 = add i64 %391, 8
  br label %390

427:                                              ; preds = %390
  %428 = add i64 %387, 8
  br label %386

429:                                              ; preds = %386
  %430 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %431 = ptrtoint ptr %430 to i64
  %432 = add i64 %431, 63
  %433 = urem i64 %432, 64
  %434 = sub i64 %432, %433
  %435 = inttoptr i64 %434 to ptr
  br label %436

436:                                              ; preds = %439, %429
  %437 = phi i64 [ %441, %439 ], [ 0, %429 ]
  %438 = icmp slt i64 %437, 1
  br i1 %438, label %439, label %442

439:                                              ; preds = %436
  %440 = getelementptr float, ptr %435, i64 %437
  store float 0xFFF0000000000000, ptr %440, align 4
  %441 = add i64 %437, 1
  br label %436

442:                                              ; preds = %436
  %443 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), i64 64))
  %444 = ptrtoint ptr %443 to i64
  %445 = add i64 %444, 63
  %446 = urem i64 %445, 64
  %447 = sub i64 %445, %446
  %448 = inttoptr i64 %447 to ptr
  %449 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %443, 0
  %450 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %449, ptr %448, 1
  %451 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %450, i64 0, 2
  %452 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %451, i64 1, 3, 0
  %453 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %452, i64 1, 4, 0
  br label %454

454:                                              ; preds = %457, %442
  %455 = phi i64 [ %459, %457 ], [ 0, %442 ]
  %456 = icmp slt i64 %455, 1
  br i1 %456, label %457, label %460

457:                                              ; preds = %454
  %458 = getelementptr i64, ptr %448, i64 %455
  store i64 0, ptr %458, align 4
  %459 = add i64 %455, 1
  br label %454

460:                                              ; preds = %454
  br label %461

461:                                              ; preds = %489, %460
  %462 = phi i64 [ %490, %489 ], [ 0, %460 ]
  %463 = icmp slt i64 %462, 32000
  br i1 %463, label %464, label %491

464:                                              ; preds = %461
  br label %465

465:                                              ; preds = %487, %464
  %466 = phi i64 [ %488, %487 ], [ 0, %464 ]
  %467 = icmp slt i64 %466, 1
  br i1 %467, label %468, label %489

468:                                              ; preds = %465
  br label %469

469:                                              ; preds = %472, %468
  %470 = phi i64 [ %486, %472 ], [ 0, %468 ]
  %471 = icmp slt i64 %470, 8
  br i1 %471, label %472, label %487

472:                                              ; preds = %469
  %473 = getelementptr float, ptr %362, i64 %462
  %474 = mul i64 %466, 32000
  %475 = add i64 %474, %470
  %476 = getelementptr float, ptr %473, i64 %475
  %477 = load float, ptr %476, align 4
  %478 = getelementptr float, ptr %435, i64 %466
  %479 = load float, ptr %478, align 4
  %480 = getelementptr i64, ptr %448, i64 %466
  %481 = load i64, ptr %480, align 4
  %482 = add i64 %470, %462
  %483 = fcmp ogt float %477, %479
  %484 = select i1 %483, float %477, float %479
  %485 = select i1 %483, i64 %482, i64 %481
  store float %484, ptr %478, align 4
  store i64 %485, ptr %480, align 4
  %486 = add i64 %470, 1
  br label %469

487:                                              ; preds = %469
  %488 = add i64 %466, 1
  br label %465

489:                                              ; preds = %465
  %490 = add i64 %462, 8
  br label %461

491:                                              ; preds = %461
  %492 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %453, ptr %492, align 8
  call void @printMemrefI64(i64 1, ptr %492)
  %493 = add i64 %17, 1
  br label %13

494:                                              ; preds = %13
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
