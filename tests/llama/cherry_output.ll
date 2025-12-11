; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__constant_12x1024x768xf32 = private constant [12 x [1024 x [768 x float]]] zeroinitializer, align 64
@__constant_1x12x64xf32 = private constant [1 x [12 x [64 x float]]] zeroinitializer, align 64
@__constant_1x768xf32 = private constant [1 x [768 x float]] [[768 x float] [float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00]], align 64
@__constant_3xi64_1 = private constant [3 x i64] [i64 1, i64 12, i64 64], align 64
@__constant_2xi64_0 = private constant [2 x i64] [i64 1, i64 768], align 64
@__constant_3xi64_0 = private constant [3 x i64] [i64 1, i64 1, i64 768], align 64
@__constant_2xi64 = private constant [2 x i64] [i64 1024, i64 768], align 64
@__constant_3xi64 = private constant [3 x i64] [i64 1, i64 1, i64 64], align 64

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

declare void @printMemrefF32(i64, ptr)

define void @host() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1, 0
  %8 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %8, i64 0, 2
  %10 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %9, i64 12, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, i64 1024, 3, 1
  %12 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %11, i64 768, 3, 2
  %13 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %12, i64 786432, 4, 0
  %14 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, i64 768, 4, 1
  %15 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @__constant_12x1024x768xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9437184), i1 false)
  %16 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %17 = ptrtoint ptr %16 to i64
  %18 = add i64 %17, 63
  %19 = urem i64 %18, 64
  %20 = sub i64 %18, %19
  %21 = inttoptr i64 %20 to ptr
  %22 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %16, 0
  %23 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, ptr %21, 1
  %24 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %23, i64 0, 2
  %25 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, i64 12, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %25, i64 1024, 3, 1
  %27 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %26, i64 768, 3, 2
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, i64 786432, 4, 0
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, i64 768, 4, 1
  %30 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %21, ptr @__constant_12x1024x768xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9437184), i1 false)
  br label %31

31:                                               ; preds = %2177, %0
  %32 = phi i64 [ %2179, %2177 ], [ 0, %0 ]
  %33 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %106, %2177 ], [ %15, %0 ]
  %34 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %107, %2177 ], [ %30, %0 ]
  %35 = icmp slt i64 %32, 10
  %36 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %37 = ptrtoint ptr %36 to i64
  %38 = add i64 %37, 63
  %39 = urem i64 %38, 64
  %40 = sub i64 %38, %39
  %41 = inttoptr i64 %40 to ptr
  %42 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %36, 0
  %43 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, ptr %41, 1
  %44 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %43, i64 0, 2
  %45 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %44, i64 12, 3, 0
  %46 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %45, i64 1024, 3, 1
  %47 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %46, i64 768, 3, 2
  %48 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %47, i64 786432, 4, 0
  %49 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %48, i64 768, 4, 1
  %50 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, i64 1, 4, 2
  %51 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3, 0
  %52 = mul i64 %51, 1
  %53 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3, 1
  %54 = mul i64 %52, %53
  %55 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3, 2
  %56 = mul i64 %54, %55
  %57 = mul i64 %56, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %58 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 1
  %59 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 2
  %60 = getelementptr float, ptr %58, i64 %59
  call void @llvm.memcpy.p0.p0.i64(ptr %41, ptr %60, i64 %57, i1 false)
  %61 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %62 = ptrtoint ptr %61 to i64
  %63 = add i64 %62, 63
  %64 = urem i64 %63, 64
  %65 = sub i64 %63, %64
  %66 = inttoptr i64 %65 to ptr
  %67 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %61, 0
  %68 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %67, ptr %66, 1
  %69 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %68, i64 0, 2
  %70 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, i64 12, 3, 0
  %71 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %70, i64 1024, 3, 1
  %72 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %71, i64 768, 3, 2
  %73 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %72, i64 786432, 4, 0
  %74 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %73, i64 768, 4, 1
  %75 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %74, i64 1, 4, 2
  %76 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, 3, 0
  %77 = mul i64 %76, 1
  %78 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, 3, 1
  %79 = mul i64 %77, %78
  %80 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, 3, 2
  %81 = mul i64 %79, %80
  %82 = mul i64 %81, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %83 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, 1
  %84 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, 2
  %85 = getelementptr float, ptr %83, i64 %84
  call void @llvm.memcpy.p0.p0.i64(ptr %66, ptr %85, i64 %82, i1 false)
  br i1 %35, label %86, label %2180

86:                                               ; preds = %31
  %87 = phi i64 [ %32, %31 ]
  %88 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %50, %31 ]
  %89 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %75, %31 ]
  %90 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %91 = ptrtoint ptr %90 to i64
  %92 = add i64 %91, 63
  %93 = urem i64 %92, 64
  %94 = sub i64 %92, %93
  %95 = inttoptr i64 %94 to ptr
  %96 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %90, 0
  %97 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, ptr %95, 1
  %98 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %97, i64 0, 2
  %99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, i64 1, 3, 0
  %100 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, i64 768, 3, 1
  %101 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %100, i64 768, 4, 0
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %101, i64 1, 4, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %95, ptr @__constant_1x768xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %103

103:                                              ; preds = %1959, %86
  %104 = phi i64 [ %1990, %1959 ], [ 0, %86 ]
  %105 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %1929, %1959 ], [ %102, %86 ]
  %106 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %1974, %1959 ], [ %88, %86 ]
  %107 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %1989, %1959 ], [ %89, %86 ]
  %108 = icmp slt i64 %104, 12
  br i1 %108, label %109, label %1991

109:                                              ; preds = %103
  %110 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %111 = ptrtoint ptr %110 to i64
  %112 = add i64 %111, 63
  %113 = urem i64 %112, 64
  %114 = sub i64 %112, %113
  %115 = inttoptr i64 %114 to ptr
  br label %116

116:                                              ; preds = %119, %109
  %117 = phi i64 [ %121, %119 ], [ 0, %109 ]
  %118 = icmp slt i64 %117, 1
  br i1 %118, label %119, label %122

119:                                              ; preds = %116
  %120 = getelementptr float, ptr %115, i64 %117
  store float 0.000000e+00, ptr %120, align 4
  %121 = add i64 %117, 1
  br label %116

122:                                              ; preds = %116
  br label %123

123:                                              ; preds = %148, %122
  %124 = phi i64 [ %149, %148 ], [ 0, %122 ]
  %125 = icmp slt i64 %124, 768
  br i1 %125, label %126, label %150

126:                                              ; preds = %123
  %127 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %105, 1
  br label %128

128:                                              ; preds = %146, %126
  %129 = phi i64 [ %147, %146 ], [ 0, %126 ]
  %130 = icmp slt i64 %129, 1
  br i1 %130, label %131, label %148

131:                                              ; preds = %128
  br label %132

132:                                              ; preds = %135, %131
  %133 = phi i64 [ %145, %135 ], [ 0, %131 ]
  %134 = icmp slt i64 %133, 8
  br i1 %134, label %135, label %146

135:                                              ; preds = %132
  %136 = getelementptr float, ptr %127, i64 %124
  %137 = mul i64 %129, 768
  %138 = add i64 %137, %133
  %139 = getelementptr float, ptr %136, i64 %138
  %140 = load float, ptr %139, align 4
  %141 = getelementptr float, ptr %115, i64 %129
  %142 = load float, ptr %141, align 4
  %143 = fmul float %140, %140
  %144 = fadd float %142, %143
  store float %144, ptr %141, align 4
  %145 = add i64 %133, 1
  br label %132

146:                                              ; preds = %132
  %147 = add i64 %129, 1
  br label %128

148:                                              ; preds = %128
  %149 = add i64 %124, 8
  br label %123

150:                                              ; preds = %123
  %151 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %152 = ptrtoint ptr %151 to i64
  %153 = add i64 %152, 63
  %154 = urem i64 %153, 64
  %155 = sub i64 %153, %154
  %156 = inttoptr i64 %155 to ptr
  br label %157

157:                                              ; preds = %188, %150
  %158 = phi i64 [ %190, %188 ], [ 0, %150 ]
  %159 = icmp slt i64 %158, 768
  br i1 %159, label %160, label %191

160:                                              ; preds = %157
  %161 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %105, 1
  br label %162

162:                                              ; preds = %186, %160
  %163 = phi i64 [ %187, %186 ], [ 0, %160 ]
  %164 = icmp slt i64 %163, 1
  br i1 %164, label %165, label %188

165:                                              ; preds = %162
  br label %166

166:                                              ; preds = %169, %165
  %167 = phi i64 [ %185, %169 ], [ 0, %165 ]
  %168 = icmp slt i64 %167, 8
  br i1 %168, label %169, label %186

169:                                              ; preds = %166
  %170 = getelementptr float, ptr %161, i64 %158
  %171 = mul i64 %163, 768
  %172 = add i64 %171, %167
  %173 = getelementptr float, ptr %170, i64 %172
  %174 = load float, ptr %173, align 4
  %175 = getelementptr float, ptr %115, i64 %163
  %176 = load float, ptr %175, align 4
  %177 = fdiv float %176, 7.680000e+02
  %178 = fadd float %177, 0x3EE4F8B580000000
  %179 = call float @llvm.sqrt.f32(float %178)
  %180 = fdiv float 1.000000e+00, %179
  %181 = fmul float %174, %180
  %182 = fmul float %181, 3.000000e+00
  %183 = getelementptr float, ptr %156, i64 %158
  %184 = getelementptr float, ptr %183, i64 %172
  store float %182, ptr %184, align 4
  %185 = add i64 %167, 1
  br label %166

186:                                              ; preds = %166
  %187 = add i64 %163, 1
  br label %162

188:                                              ; preds = %162
  %189 = getelementptr float, ptr %156, i64 %158
  call void @llvm.memcpy.p0.p0.i64(ptr %189, ptr %189, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %190 = add i64 %158, 8
  br label %157

191:                                              ; preds = %157
  %192 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %193 = ptrtoint ptr %192 to i64
  %194 = add i64 %193, 63
  %195 = urem i64 %194, 64
  %196 = sub i64 %194, %195
  %197 = inttoptr i64 %196 to ptr
  br label %198

198:                                              ; preds = %217, %191
  %199 = phi i64 [ %219, %217 ], [ 0, %191 ]
  %200 = icmp slt i64 %199, 768
  br i1 %200, label %201, label %220

201:                                              ; preds = %198
  br label %202

202:                                              ; preds = %215, %201
  %203 = phi i64 [ %216, %215 ], [ 0, %201 ]
  %204 = icmp slt i64 %203, 1
  br i1 %204, label %205, label %217

205:                                              ; preds = %202
  br label %206

206:                                              ; preds = %209, %205
  %207 = phi i64 [ %214, %209 ], [ 0, %205 ]
  %208 = icmp slt i64 %207, 8
  br i1 %208, label %209, label %215

209:                                              ; preds = %206
  %210 = getelementptr float, ptr %197, i64 %199
  %211 = mul i64 %203, 768
  %212 = add i64 %211, %207
  %213 = getelementptr float, ptr %210, i64 %212
  store float 0.000000e+00, ptr %213, align 4
  %214 = add i64 %207, 1
  br label %206

215:                                              ; preds = %206
  %216 = add i64 %203, 1
  br label %202

217:                                              ; preds = %202
  %218 = getelementptr float, ptr %197, i64 %199
  call void @llvm.memcpy.p0.p0.i64(ptr %218, ptr %218, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %219 = add i64 %199, 8
  br label %198

220:                                              ; preds = %198
  %221 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %222 = ptrtoint ptr %221 to i64
  %223 = add i64 %222, 63
  %224 = urem i64 %223, 64
  %225 = sub i64 %223, %224
  %226 = inttoptr i64 %225 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %226, ptr %197, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %227

227:                                              ; preds = %266, %220
  %228 = phi i64 [ %267, %266 ], [ 0, %220 ]
  %229 = icmp slt i64 %228, 768
  br i1 %229, label %230, label %268

230:                                              ; preds = %227
  br label %231

231:                                              ; preds = %263, %230
  %232 = phi i64 [ %265, %263 ], [ 0, %230 ]
  %233 = icmp slt i64 %232, 768
  br i1 %233, label %234, label %266

234:                                              ; preds = %231
  br label %235

235:                                              ; preds = %261, %234
  %236 = phi i64 [ %262, %261 ], [ 0, %234 ]
  %237 = icmp slt i64 %236, 1
  br i1 %237, label %238, label %263

238:                                              ; preds = %235
  br label %239

239:                                              ; preds = %259, %238
  %240 = phi i64 [ %260, %259 ], [ 0, %238 ]
  %241 = icmp slt i64 %240, 8
  br i1 %241, label %242, label %261

242:                                              ; preds = %239
  br label %243

243:                                              ; preds = %246, %242
  %244 = phi i64 [ %258, %246 ], [ 0, %242 ]
  %245 = icmp slt i64 %244, 8
  br i1 %245, label %246, label %259

246:                                              ; preds = %243
  %247 = getelementptr float, ptr %156, i64 %232
  %248 = mul i64 %236, 768
  %249 = add i64 %248, %244
  %250 = getelementptr float, ptr %247, i64 %249
  %251 = load float, ptr %250, align 4
  %252 = getelementptr float, ptr %226, i64 %228
  %253 = add i64 %248, %240
  %254 = getelementptr float, ptr %252, i64 %253
  %255 = load float, ptr %254, align 4
  %256 = fmul float %251, 4.000000e+00
  %257 = fadd float %255, %256
  store float %257, ptr %254, align 4
  %258 = add i64 %244, 1
  br label %243

259:                                              ; preds = %243
  %260 = add i64 %240, 1
  br label %239

261:                                              ; preds = %239
  %262 = add i64 %236, 1
  br label %235

263:                                              ; preds = %235
  %264 = getelementptr float, ptr %226, i64 %228
  call void @llvm.memcpy.p0.p0.i64(ptr %264, ptr %264, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %265 = add i64 %232, 8
  br label %231

266:                                              ; preds = %231
  %267 = add i64 %228, 8
  br label %227

268:                                              ; preds = %227
  %269 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %270 = ptrtoint ptr %269 to i64
  %271 = add i64 %270, 63
  %272 = urem i64 %271, 64
  %273 = sub i64 %271, %272
  %274 = inttoptr i64 %273 to ptr
  br label %275

275:                                              ; preds = %294, %268
  %276 = phi i64 [ %296, %294 ], [ 0, %268 ]
  %277 = icmp slt i64 %276, 768
  br i1 %277, label %278, label %297

278:                                              ; preds = %275
  br label %279

279:                                              ; preds = %292, %278
  %280 = phi i64 [ %293, %292 ], [ 0, %278 ]
  %281 = icmp slt i64 %280, 1
  br i1 %281, label %282, label %294

282:                                              ; preds = %279
  br label %283

283:                                              ; preds = %286, %282
  %284 = phi i64 [ %291, %286 ], [ 0, %282 ]
  %285 = icmp slt i64 %284, 8
  br i1 %285, label %286, label %292

286:                                              ; preds = %283
  %287 = getelementptr float, ptr %274, i64 %276
  %288 = mul i64 %280, 768
  %289 = add i64 %288, %284
  %290 = getelementptr float, ptr %287, i64 %289
  store float 0.000000e+00, ptr %290, align 4
  %291 = add i64 %284, 1
  br label %283

292:                                              ; preds = %283
  %293 = add i64 %280, 1
  br label %279

294:                                              ; preds = %279
  %295 = getelementptr float, ptr %274, i64 %276
  call void @llvm.memcpy.p0.p0.i64(ptr %295, ptr %295, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %296 = add i64 %276, 8
  br label %275

297:                                              ; preds = %275
  %298 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %299 = ptrtoint ptr %298 to i64
  %300 = add i64 %299, 63
  %301 = urem i64 %300, 64
  %302 = sub i64 %300, %301
  %303 = inttoptr i64 %302 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %303, ptr %274, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %304

304:                                              ; preds = %343, %297
  %305 = phi i64 [ %344, %343 ], [ 0, %297 ]
  %306 = icmp slt i64 %305, 768
  br i1 %306, label %307, label %345

307:                                              ; preds = %304
  br label %308

308:                                              ; preds = %340, %307
  %309 = phi i64 [ %342, %340 ], [ 0, %307 ]
  %310 = icmp slt i64 %309, 768
  br i1 %310, label %311, label %343

311:                                              ; preds = %308
  br label %312

312:                                              ; preds = %338, %311
  %313 = phi i64 [ %339, %338 ], [ 0, %311 ]
  %314 = icmp slt i64 %313, 1
  br i1 %314, label %315, label %340

315:                                              ; preds = %312
  br label %316

316:                                              ; preds = %336, %315
  %317 = phi i64 [ %337, %336 ], [ 0, %315 ]
  %318 = icmp slt i64 %317, 8
  br i1 %318, label %319, label %338

319:                                              ; preds = %316
  br label %320

320:                                              ; preds = %323, %319
  %321 = phi i64 [ %335, %323 ], [ 0, %319 ]
  %322 = icmp slt i64 %321, 8
  br i1 %322, label %323, label %336

323:                                              ; preds = %320
  %324 = getelementptr float, ptr %156, i64 %309
  %325 = mul i64 %313, 768
  %326 = add i64 %325, %321
  %327 = getelementptr float, ptr %324, i64 %326
  %328 = load float, ptr %327, align 4
  %329 = getelementptr float, ptr %303, i64 %305
  %330 = add i64 %325, %317
  %331 = getelementptr float, ptr %329, i64 %330
  %332 = load float, ptr %331, align 4
  %333 = fmul float %328, 5.000000e+00
  %334 = fadd float %332, %333
  store float %334, ptr %331, align 4
  %335 = add i64 %321, 1
  br label %320

336:                                              ; preds = %320
  %337 = add i64 %317, 1
  br label %316

338:                                              ; preds = %316
  %339 = add i64 %313, 1
  br label %312

340:                                              ; preds = %312
  %341 = getelementptr float, ptr %303, i64 %305
  call void @llvm.memcpy.p0.p0.i64(ptr %341, ptr %341, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %342 = add i64 %309, 8
  br label %308

343:                                              ; preds = %308
  %344 = add i64 %305, 8
  br label %304

345:                                              ; preds = %304
  %346 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %347 = ptrtoint ptr %346 to i64
  %348 = add i64 %347, 63
  %349 = urem i64 %348, 64
  %350 = sub i64 %348, %349
  %351 = inttoptr i64 %350 to ptr
  br label %352

352:                                              ; preds = %371, %345
  %353 = phi i64 [ %373, %371 ], [ 0, %345 ]
  %354 = icmp slt i64 %353, 768
  br i1 %354, label %355, label %374

355:                                              ; preds = %352
  br label %356

356:                                              ; preds = %369, %355
  %357 = phi i64 [ %370, %369 ], [ 0, %355 ]
  %358 = icmp slt i64 %357, 1
  br i1 %358, label %359, label %371

359:                                              ; preds = %356
  br label %360

360:                                              ; preds = %363, %359
  %361 = phi i64 [ %368, %363 ], [ 0, %359 ]
  %362 = icmp slt i64 %361, 8
  br i1 %362, label %363, label %369

363:                                              ; preds = %360
  %364 = getelementptr float, ptr %351, i64 %353
  %365 = mul i64 %357, 768
  %366 = add i64 %365, %361
  %367 = getelementptr float, ptr %364, i64 %366
  store float 0.000000e+00, ptr %367, align 4
  %368 = add i64 %361, 1
  br label %360

369:                                              ; preds = %360
  %370 = add i64 %357, 1
  br label %356

371:                                              ; preds = %356
  %372 = getelementptr float, ptr %351, i64 %353
  call void @llvm.memcpy.p0.p0.i64(ptr %372, ptr %372, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %373 = add i64 %353, 8
  br label %352

374:                                              ; preds = %352
  %375 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %376 = ptrtoint ptr %375 to i64
  %377 = add i64 %376, 63
  %378 = urem i64 %377, 64
  %379 = sub i64 %377, %378
  %380 = inttoptr i64 %379 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %380, ptr %351, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %381

381:                                              ; preds = %420, %374
  %382 = phi i64 [ %421, %420 ], [ 0, %374 ]
  %383 = icmp slt i64 %382, 768
  br i1 %383, label %384, label %422

384:                                              ; preds = %381
  br label %385

385:                                              ; preds = %417, %384
  %386 = phi i64 [ %419, %417 ], [ 0, %384 ]
  %387 = icmp slt i64 %386, 768
  br i1 %387, label %388, label %420

388:                                              ; preds = %385
  br label %389

389:                                              ; preds = %415, %388
  %390 = phi i64 [ %416, %415 ], [ 0, %388 ]
  %391 = icmp slt i64 %390, 1
  br i1 %391, label %392, label %417

392:                                              ; preds = %389
  br label %393

393:                                              ; preds = %413, %392
  %394 = phi i64 [ %414, %413 ], [ 0, %392 ]
  %395 = icmp slt i64 %394, 8
  br i1 %395, label %396, label %415

396:                                              ; preds = %393
  br label %397

397:                                              ; preds = %400, %396
  %398 = phi i64 [ %412, %400 ], [ 0, %396 ]
  %399 = icmp slt i64 %398, 8
  br i1 %399, label %400, label %413

400:                                              ; preds = %397
  %401 = getelementptr float, ptr %156, i64 %386
  %402 = mul i64 %390, 768
  %403 = add i64 %402, %398
  %404 = getelementptr float, ptr %401, i64 %403
  %405 = load float, ptr %404, align 4
  %406 = getelementptr float, ptr %380, i64 %382
  %407 = add i64 %402, %394
  %408 = getelementptr float, ptr %406, i64 %407
  %409 = load float, ptr %408, align 4
  %410 = fmul float %405, 6.000000e+00
  %411 = fadd float %409, %410
  store float %411, ptr %408, align 4
  %412 = add i64 %398, 1
  br label %397

413:                                              ; preds = %397
  %414 = add i64 %394, 1
  br label %393

415:                                              ; preds = %393
  %416 = add i64 %390, 1
  br label %389

417:                                              ; preds = %389
  %418 = getelementptr float, ptr %380, i64 %382
  call void @llvm.memcpy.p0.p0.i64(ptr %418, ptr %418, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %419 = add i64 %386, 8
  br label %385

420:                                              ; preds = %385
  %421 = add i64 %382, 8
  br label %381

422:                                              ; preds = %381
  %423 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %424 = ptrtoint ptr %423 to i64
  %425 = add i64 %424, 63
  %426 = urem i64 %425, 64
  %427 = sub i64 %425, %426
  %428 = inttoptr i64 %427 to ptr
  %429 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %430 = ptrtoint ptr %429 to i64
  %431 = add i64 %430, 63
  %432 = urem i64 %431, 64
  %433 = sub i64 %431, %432
  %434 = inttoptr i64 %433 to ptr
  %435 = uitofp i64 %87 to float
  br label %436

436:                                              ; preds = %457, %422
  %437 = phi i64 [ %460, %457 ], [ 0, %422 ]
  %438 = icmp slt i64 %437, 32
  br i1 %438, label %439, label %461

439:                                              ; preds = %436
  br label %440

440:                                              ; preds = %443, %439
  %441 = phi i64 [ %456, %443 ], [ 0, %439 ]
  %442 = icmp slt i64 %441, 8
  br i1 %442, label %443, label %457

443:                                              ; preds = %440
  %444 = add i64 %441, %437
  %445 = uitofp i64 %444 to float
  %446 = fmul float %445, -2.000000e+00
  %447 = fdiv float %446, 6.400000e+01
  %448 = call float @llvm.pow.f32(float 1.000000e+04, float %447)
  %449 = fmul float %435, %448
  %450 = call float @llvm.cos.f32(float %449)
  %451 = call float @llvm.sin.f32(float %449)
  %452 = getelementptr float, ptr %428, i64 %437
  %453 = getelementptr float, ptr %452, i64 %441
  store float %450, ptr %453, align 4
  %454 = getelementptr float, ptr %434, i64 %437
  %455 = getelementptr float, ptr %454, i64 %441
  store float %451, ptr %455, align 4
  %456 = add i64 %441, 1
  br label %440

457:                                              ; preds = %440
  %458 = getelementptr float, ptr %428, i64 %437
  call void @llvm.memcpy.p0.p0.i64(ptr %458, ptr %458, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %459 = getelementptr float, ptr %434, i64 %437
  call void @llvm.memcpy.p0.p0.i64(ptr %459, ptr %459, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %460 = add i64 %437, 8
  br label %436

461:                                              ; preds = %436
  %462 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %463 = ptrtoint ptr %462 to i64
  %464 = add i64 %463, 63
  %465 = urem i64 %464, 64
  %466 = sub i64 %464, %465
  %467 = inttoptr i64 %466 to ptr
  %468 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %462, 0
  %469 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %468, ptr %467, 1
  %470 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %469, i64 0, 2
  %471 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %470, i64 1, 3, 0
  %472 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %471, i64 12, 3, 1
  %473 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %472, i64 32, 3, 2
  %474 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %473, i64 1, 3, 3
  %475 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %474, i64 384, 4, 0
  %476 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %475, i64 32, 4, 1
  %477 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %476, i64 1, 4, 2
  %478 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %477, i64 1, 4, 3
  %479 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %480 = ptrtoint ptr %479 to i64
  %481 = add i64 %480, 63
  %482 = urem i64 %481, 64
  %483 = sub i64 %481, %482
  %484 = inttoptr i64 %483 to ptr
  %485 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %479, 0
  %486 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %485, ptr %484, 1
  %487 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %486, i64 0, 2
  %488 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %487, i64 1, 3, 0
  %489 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %488, i64 12, 3, 1
  %490 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %489, i64 32, 3, 2
  %491 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %490, i64 1, 3, 3
  %492 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %491, i64 384, 4, 0
  %493 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %492, i64 32, 4, 1
  %494 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %493, i64 1, 4, 2
  %495 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %494, i64 1, 4, 3
  br label %496

496:                                              ; preds = %604, %461
  %497 = phi i64 [ %605, %604 ], [ 0, %461 ]
  %498 = icmp slt i64 %497, 12
  br i1 %498, label %499, label %606

499:                                              ; preds = %496
  br label %500

500:                                              ; preds = %588, %499
  %501 = phi i64 [ %603, %588 ], [ 0, %499 ]
  %502 = icmp slt i64 %501, 32
  br i1 %502, label %503, label %604

503:                                              ; preds = %500
  %504 = mul i64 %497, -1
  %505 = add i64 %504, 12
  %506 = call i64 @llvm.smin.i64(i64 %505, i64 8)
  %507 = mul i64 %497, 64
  %508 = mul i64 %501, 2
  %509 = add i64 %507, %508
  %510 = add i64 %509, 1
  %511 = mul i64 %497, 32
  %512 = add i64 %511, %501
  %513 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %469, i64 %512, 2
  %514 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %513, i64 1, 3, 0
  %515 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %514, i64 384, 4, 0
  %516 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %515, i64 %506, 3, 1
  %517 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %516, i64 32, 4, 1
  %518 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %517, i64 8, 3, 2
  %519 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %518, i64 1, 4, 2
  %520 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %519, i64 1, 3, 3
  %521 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %520, i64 1, 4, 3
  %522 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %486, i64 %512, 2
  %523 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %522, i64 1, 3, 0
  %524 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %523, i64 384, 4, 0
  %525 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %524, i64 %506, 3, 1
  %526 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %525, i64 32, 4, 1
  %527 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %526, i64 8, 3, 2
  %528 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %527, i64 1, 4, 2
  %529 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %528, i64 1, 3, 3
  %530 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %529, i64 1, 4, 3
  br label %531

531:                                              ; preds = %586, %503
  %532 = phi i64 [ %587, %586 ], [ 0, %503 ]
  %533 = icmp slt i64 %532, 1
  br i1 %533, label %534, label %588

534:                                              ; preds = %531
  br label %535

535:                                              ; preds = %584, %534
  %536 = phi i64 [ %585, %584 ], [ 0, %534 ]
  %537 = icmp slt i64 %536, %506
  br i1 %537, label %538, label %586

538:                                              ; preds = %535
  br label %539

539:                                              ; preds = %582, %538
  %540 = phi i64 [ %583, %582 ], [ 0, %538 ]
  %541 = icmp slt i64 %540, 8
  br i1 %541, label %542, label %584

542:                                              ; preds = %539
  br label %543

543:                                              ; preds = %546, %542
  %544 = phi i64 [ %581, %546 ], [ 0, %542 ]
  %545 = icmp slt i64 %544, 1
  br i1 %545, label %546, label %582

546:                                              ; preds = %543
  %547 = getelementptr float, ptr %226, i64 %509
  %548 = mul i64 %532, 768
  %549 = mul i64 %536, 64
  %550 = add i64 %548, %549
  %551 = mul i64 %540, 2
  %552 = add i64 %550, %551
  %553 = add i64 %552, %544
  %554 = getelementptr float, ptr %547, i64 %553
  %555 = load float, ptr %554, align 4
  %556 = getelementptr float, ptr %226, i64 %510
  %557 = getelementptr float, ptr %556, i64 %553
  %558 = load float, ptr %557, align 4
  %559 = getelementptr float, ptr %428, i64 %501
  %560 = add i64 %540, %544
  %561 = getelementptr float, ptr %559, i64 %560
  %562 = load float, ptr %561, align 4
  %563 = getelementptr float, ptr %434, i64 %501
  %564 = getelementptr float, ptr %563, i64 %560
  %565 = load float, ptr %564, align 4
  %566 = fmul float %555, %562
  %567 = fmul float %558, %565
  %568 = fsub float %566, %567
  %569 = fmul float %558, %562
  %570 = fmul float %555, %565
  %571 = fadd float %569, %570
  %572 = getelementptr float, ptr %467, i64 %512
  %573 = mul i64 %532, 384
  %574 = mul i64 %536, 32
  %575 = add i64 %573, %574
  %576 = add i64 %575, %540
  %577 = add i64 %576, %544
  %578 = getelementptr float, ptr %572, i64 %577
  store float %568, ptr %578, align 4
  %579 = getelementptr float, ptr %484, i64 %512
  %580 = getelementptr float, ptr %579, i64 %577
  store float %571, ptr %580, align 4
  %581 = add i64 %544, 1
  br label %543

582:                                              ; preds = %543
  %583 = add i64 %540, 1
  br label %539

584:                                              ; preds = %539
  %585 = add i64 %536, 1
  br label %535

586:                                              ; preds = %535
  %587 = add i64 %532, 1
  br label %531

588:                                              ; preds = %531
  %589 = call ptr @llvm.stacksave.p0()
  %590 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %521, ptr %590, align 8
  %591 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %590, 1
  %592 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %521, ptr %592, align 8
  %593 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %592, 1
  %594 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %591, ptr %594, align 8
  %595 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %593, ptr %595, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %594, ptr %595)
  call void @llvm.stackrestore.p0(ptr %589)
  %596 = call ptr @llvm.stacksave.p0()
  %597 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %530, ptr %597, align 8
  %598 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %597, 1
  %599 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %530, ptr %599, align 8
  %600 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %599, 1
  %601 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %598, ptr %601, align 8
  %602 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %600, ptr %602, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %601, ptr %602)
  call void @llvm.stackrestore.p0(ptr %596)
  %603 = add i64 %501, 8
  br label %500

604:                                              ; preds = %500
  %605 = add i64 %497, 8
  br label %496

606:                                              ; preds = %496
  %607 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %608 = ptrtoint ptr %607 to i64
  %609 = add i64 %608, 63
  %610 = urem i64 %609, 64
  %611 = sub i64 %609, %610
  %612 = inttoptr i64 %611 to ptr
  %613 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %607, 0
  %614 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %613, ptr %612, 1
  %615 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %614, i64 0, 2
  %616 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %615, i64 1, 3, 0
  %617 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %616, i64 768, 4, 0
  %618 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %617, i64 12, 3, 1
  %619 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %618, i64 64, 4, 1
  %620 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %619, i64 32, 3, 2
  %621 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %620, i64 2, 4, 2
  %622 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %621, i64 1, 3, 3
  %623 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %622, i64 1, 4, 3
  %624 = call ptr @llvm.stacksave.p0()
  %625 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %478, ptr %625, align 8
  %626 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %625, 1
  %627 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %623, ptr %627, align 8
  %628 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %627, 1
  %629 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %626, ptr %629, align 8
  %630 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %628, ptr %630, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %629, ptr %630)
  call void @llvm.stackrestore.p0(ptr %624)
  %631 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %632 = ptrtoint ptr %631 to i64
  %633 = add i64 %632, 63
  %634 = urem i64 %633, 64
  %635 = sub i64 %633, %634
  %636 = inttoptr i64 %635 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %636, ptr %612, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %637 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %631, 0
  %638 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %637, ptr %636, 1
  %639 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %638, i64 1, 2
  %640 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %639, i64 1, 3, 0
  %641 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %640, i64 768, 4, 0
  %642 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %641, i64 12, 3, 1
  %643 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %642, i64 64, 4, 1
  %644 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %643, i64 32, 3, 2
  %645 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %644, i64 2, 4, 2
  %646 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %645, i64 1, 3, 3
  %647 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %646, i64 1, 4, 3
  %648 = call ptr @llvm.stacksave.p0()
  %649 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %495, ptr %649, align 8
  %650 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %649, 1
  %651 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %647, ptr %651, align 8
  %652 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %651, 1
  %653 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %650, ptr %653, align 8
  %654 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %652, ptr %654, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %653, ptr %654)
  call void @llvm.stackrestore.p0(ptr %648)
  %655 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %656 = ptrtoint ptr %655 to i64
  %657 = add i64 %656, 63
  %658 = urem i64 %657, 64
  %659 = sub i64 %657, %658
  %660 = inttoptr i64 %659 to ptr
  %661 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %662 = ptrtoint ptr %661 to i64
  %663 = add i64 %662, 63
  %664 = urem i64 %663, 64
  %665 = sub i64 %663, %664
  %666 = inttoptr i64 %665 to ptr
  br label %667

667:                                              ; preds = %688, %606
  %668 = phi i64 [ %691, %688 ], [ 0, %606 ]
  %669 = icmp slt i64 %668, 32
  br i1 %669, label %670, label %692

670:                                              ; preds = %667
  br label %671

671:                                              ; preds = %674, %670
  %672 = phi i64 [ %687, %674 ], [ 0, %670 ]
  %673 = icmp slt i64 %672, 8
  br i1 %673, label %674, label %688

674:                                              ; preds = %671
  %675 = add i64 %672, %668
  %676 = uitofp i64 %675 to float
  %677 = fmul float %676, -2.000000e+00
  %678 = fdiv float %677, 6.400000e+01
  %679 = call float @llvm.pow.f32(float 1.000000e+04, float %678)
  %680 = fmul float %435, %679
  %681 = call float @llvm.cos.f32(float %680)
  %682 = call float @llvm.sin.f32(float %680)
  %683 = getelementptr float, ptr %660, i64 %668
  %684 = getelementptr float, ptr %683, i64 %672
  store float %681, ptr %684, align 4
  %685 = getelementptr float, ptr %666, i64 %668
  %686 = getelementptr float, ptr %685, i64 %672
  store float %682, ptr %686, align 4
  %687 = add i64 %672, 1
  br label %671

688:                                              ; preds = %671
  %689 = getelementptr float, ptr %660, i64 %668
  call void @llvm.memcpy.p0.p0.i64(ptr %689, ptr %689, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %690 = getelementptr float, ptr %666, i64 %668
  call void @llvm.memcpy.p0.p0.i64(ptr %690, ptr %690, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %691 = add i64 %668, 8
  br label %667

692:                                              ; preds = %667
  %693 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %694 = ptrtoint ptr %693 to i64
  %695 = add i64 %694, 63
  %696 = urem i64 %695, 64
  %697 = sub i64 %695, %696
  %698 = inttoptr i64 %697 to ptr
  %699 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %693, 0
  %700 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %699, ptr %698, 1
  %701 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %700, i64 0, 2
  %702 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %701, i64 1, 3, 0
  %703 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %702, i64 12, 3, 1
  %704 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %703, i64 32, 3, 2
  %705 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %704, i64 1, 3, 3
  %706 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %705, i64 384, 4, 0
  %707 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %706, i64 32, 4, 1
  %708 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %707, i64 1, 4, 2
  %709 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %708, i64 1, 4, 3
  %710 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %711 = ptrtoint ptr %710 to i64
  %712 = add i64 %711, 63
  %713 = urem i64 %712, 64
  %714 = sub i64 %712, %713
  %715 = inttoptr i64 %714 to ptr
  %716 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %710, 0
  %717 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %716, ptr %715, 1
  %718 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %717, i64 0, 2
  %719 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %718, i64 1, 3, 0
  %720 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %719, i64 12, 3, 1
  %721 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %720, i64 32, 3, 2
  %722 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %721, i64 1, 3, 3
  %723 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %722, i64 384, 4, 0
  %724 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %723, i64 32, 4, 1
  %725 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %724, i64 1, 4, 2
  %726 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %725, i64 1, 4, 3
  br label %727

727:                                              ; preds = %835, %692
  %728 = phi i64 [ %836, %835 ], [ 0, %692 ]
  %729 = icmp slt i64 %728, 12
  br i1 %729, label %730, label %837

730:                                              ; preds = %727
  br label %731

731:                                              ; preds = %819, %730
  %732 = phi i64 [ %834, %819 ], [ 0, %730 ]
  %733 = icmp slt i64 %732, 32
  br i1 %733, label %734, label %835

734:                                              ; preds = %731
  %735 = mul i64 %728, -1
  %736 = add i64 %735, 12
  %737 = call i64 @llvm.smin.i64(i64 %736, i64 8)
  %738 = mul i64 %728, 64
  %739 = mul i64 %732, 2
  %740 = add i64 %738, %739
  %741 = add i64 %740, 1
  %742 = mul i64 %728, 32
  %743 = add i64 %742, %732
  %744 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %700, i64 %743, 2
  %745 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %744, i64 1, 3, 0
  %746 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %745, i64 384, 4, 0
  %747 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %746, i64 %737, 3, 1
  %748 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %747, i64 32, 4, 1
  %749 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %748, i64 8, 3, 2
  %750 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %749, i64 1, 4, 2
  %751 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %750, i64 1, 3, 3
  %752 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %751, i64 1, 4, 3
  %753 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %717, i64 %743, 2
  %754 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %753, i64 1, 3, 0
  %755 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %754, i64 384, 4, 0
  %756 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %755, i64 %737, 3, 1
  %757 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %756, i64 32, 4, 1
  %758 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %757, i64 8, 3, 2
  %759 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %758, i64 1, 4, 2
  %760 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %759, i64 1, 3, 3
  %761 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %760, i64 1, 4, 3
  br label %762

762:                                              ; preds = %817, %734
  %763 = phi i64 [ %818, %817 ], [ 0, %734 ]
  %764 = icmp slt i64 %763, 1
  br i1 %764, label %765, label %819

765:                                              ; preds = %762
  br label %766

766:                                              ; preds = %815, %765
  %767 = phi i64 [ %816, %815 ], [ 0, %765 ]
  %768 = icmp slt i64 %767, %737
  br i1 %768, label %769, label %817

769:                                              ; preds = %766
  br label %770

770:                                              ; preds = %813, %769
  %771 = phi i64 [ %814, %813 ], [ 0, %769 ]
  %772 = icmp slt i64 %771, 8
  br i1 %772, label %773, label %815

773:                                              ; preds = %770
  br label %774

774:                                              ; preds = %777, %773
  %775 = phi i64 [ %812, %777 ], [ 0, %773 ]
  %776 = icmp slt i64 %775, 1
  br i1 %776, label %777, label %813

777:                                              ; preds = %774
  %778 = getelementptr float, ptr %303, i64 %740
  %779 = mul i64 %763, 768
  %780 = mul i64 %767, 64
  %781 = add i64 %779, %780
  %782 = mul i64 %771, 2
  %783 = add i64 %781, %782
  %784 = add i64 %783, %775
  %785 = getelementptr float, ptr %778, i64 %784
  %786 = load float, ptr %785, align 4
  %787 = getelementptr float, ptr %303, i64 %741
  %788 = getelementptr float, ptr %787, i64 %784
  %789 = load float, ptr %788, align 4
  %790 = getelementptr float, ptr %660, i64 %732
  %791 = add i64 %771, %775
  %792 = getelementptr float, ptr %790, i64 %791
  %793 = load float, ptr %792, align 4
  %794 = getelementptr float, ptr %666, i64 %732
  %795 = getelementptr float, ptr %794, i64 %791
  %796 = load float, ptr %795, align 4
  %797 = fmul float %786, %793
  %798 = fmul float %789, %796
  %799 = fsub float %797, %798
  %800 = fmul float %789, %793
  %801 = fmul float %786, %796
  %802 = fadd float %800, %801
  %803 = getelementptr float, ptr %698, i64 %743
  %804 = mul i64 %763, 384
  %805 = mul i64 %767, 32
  %806 = add i64 %804, %805
  %807 = add i64 %806, %771
  %808 = add i64 %807, %775
  %809 = getelementptr float, ptr %803, i64 %808
  store float %799, ptr %809, align 4
  %810 = getelementptr float, ptr %715, i64 %743
  %811 = getelementptr float, ptr %810, i64 %808
  store float %802, ptr %811, align 4
  %812 = add i64 %775, 1
  br label %774

813:                                              ; preds = %774
  %814 = add i64 %771, 1
  br label %770

815:                                              ; preds = %770
  %816 = add i64 %767, 1
  br label %766

817:                                              ; preds = %766
  %818 = add i64 %763, 1
  br label %762

819:                                              ; preds = %762
  %820 = call ptr @llvm.stacksave.p0()
  %821 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %752, ptr %821, align 8
  %822 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %821, 1
  %823 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %752, ptr %823, align 8
  %824 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %823, 1
  %825 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %822, ptr %825, align 8
  %826 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %824, ptr %826, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %825, ptr %826)
  call void @llvm.stackrestore.p0(ptr %820)
  %827 = call ptr @llvm.stacksave.p0()
  %828 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %761, ptr %828, align 8
  %829 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %828, 1
  %830 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %761, ptr %830, align 8
  %831 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %830, 1
  %832 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %829, ptr %832, align 8
  %833 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %831, ptr %833, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %832, ptr %833)
  call void @llvm.stackrestore.p0(ptr %827)
  %834 = add i64 %732, 8
  br label %731

835:                                              ; preds = %731
  %836 = add i64 %728, 8
  br label %727

837:                                              ; preds = %727
  %838 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %839 = ptrtoint ptr %838 to i64
  %840 = add i64 %839, 63
  %841 = urem i64 %840, 64
  %842 = sub i64 %840, %841
  %843 = inttoptr i64 %842 to ptr
  %844 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %838, 0
  %845 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %844, ptr %843, 1
  %846 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %845, i64 0, 2
  %847 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %846, i64 1, 3, 0
  %848 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %847, i64 768, 4, 0
  %849 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %848, i64 12, 3, 1
  %850 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %849, i64 64, 4, 1
  %851 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %850, i64 32, 3, 2
  %852 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %851, i64 2, 4, 2
  %853 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %852, i64 1, 3, 3
  %854 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %853, i64 1, 4, 3
  %855 = call ptr @llvm.stacksave.p0()
  %856 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %709, ptr %856, align 8
  %857 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %856, 1
  %858 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %854, ptr %858, align 8
  %859 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %858, 1
  %860 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %857, ptr %860, align 8
  %861 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %859, ptr %861, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %860, ptr %861)
  call void @llvm.stackrestore.p0(ptr %855)
  %862 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %863 = ptrtoint ptr %862 to i64
  %864 = add i64 %863, 63
  %865 = urem i64 %864, 64
  %866 = sub i64 %864, %865
  %867 = inttoptr i64 %866 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %867, ptr %843, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %868 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %862, 0
  %869 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %868, ptr %867, 1
  %870 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %869, i64 1, 2
  %871 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %870, i64 1, 3, 0
  %872 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %871, i64 768, 4, 0
  %873 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %872, i64 12, 3, 1
  %874 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %873, i64 64, 4, 1
  %875 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %874, i64 32, 3, 2
  %876 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %875, i64 2, 4, 2
  %877 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %876, i64 1, 3, 3
  %878 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %877, i64 1, 4, 3
  %879 = call ptr @llvm.stacksave.p0()
  %880 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %726, ptr %880, align 8
  %881 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %880, 1
  %882 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %878, ptr %882, align 8
  %883 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %882, 1
  %884 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %881, ptr %884, align 8
  %885 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %883, ptr %885, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %884, ptr %885)
  call void @llvm.stackrestore.p0(ptr %879)
  %886 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %887 = ptrtoint ptr %886 to i64
  %888 = add i64 %887, 63
  %889 = urem i64 %888, 64
  %890 = sub i64 %888, %889
  %891 = inttoptr i64 %890 to ptr
  %892 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32) to i64), i64 64))
  %893 = ptrtoint ptr %892 to i64
  %894 = add i64 %893, 63
  %895 = urem i64 %894, 64
  %896 = sub i64 %894, %895
  %897 = inttoptr i64 %896 to ptr
  br label %898

898:                                              ; preds = %919, %837
  %899 = phi i64 [ %922, %919 ], [ 0, %837 ]
  %900 = icmp slt i64 %899, 32
  br i1 %900, label %901, label %923

901:                                              ; preds = %898
  br label %902

902:                                              ; preds = %905, %901
  %903 = phi i64 [ %918, %905 ], [ 0, %901 ]
  %904 = icmp slt i64 %903, 8
  br i1 %904, label %905, label %919

905:                                              ; preds = %902
  %906 = add i64 %903, %899
  %907 = uitofp i64 %906 to float
  %908 = fmul float %907, -2.000000e+00
  %909 = fdiv float %908, 6.400000e+01
  %910 = call float @llvm.pow.f32(float 1.000000e+04, float %909)
  %911 = fmul float %435, %910
  %912 = call float @llvm.cos.f32(float %911)
  %913 = call float @llvm.sin.f32(float %911)
  %914 = getelementptr float, ptr %891, i64 %899
  %915 = getelementptr float, ptr %914, i64 %903
  store float %912, ptr %915, align 4
  %916 = getelementptr float, ptr %897, i64 %899
  %917 = getelementptr float, ptr %916, i64 %903
  store float %913, ptr %917, align 4
  %918 = add i64 %903, 1
  br label %902

919:                                              ; preds = %902
  %920 = getelementptr float, ptr %891, i64 %899
  call void @llvm.memcpy.p0.p0.i64(ptr %920, ptr %920, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %921 = getelementptr float, ptr %897, i64 %899
  call void @llvm.memcpy.p0.p0.i64(ptr %921, ptr %921, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %922 = add i64 %899, 8
  br label %898

923:                                              ; preds = %898
  %924 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %925 = ptrtoint ptr %924 to i64
  %926 = add i64 %925, 63
  %927 = urem i64 %926, 64
  %928 = sub i64 %926, %927
  %929 = inttoptr i64 %928 to ptr
  %930 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %924, 0
  %931 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %930, ptr %929, 1
  %932 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %931, i64 0, 2
  %933 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %932, i64 1, 3, 0
  %934 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %933, i64 12, 3, 1
  %935 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %934, i64 32, 3, 2
  %936 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %935, i64 1, 3, 3
  %937 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %936, i64 384, 4, 0
  %938 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %937, i64 32, 4, 1
  %939 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %938, i64 1, 4, 2
  %940 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %939, i64 1, 4, 3
  %941 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 384) to i64), i64 64))
  %942 = ptrtoint ptr %941 to i64
  %943 = add i64 %942, 63
  %944 = urem i64 %943, 64
  %945 = sub i64 %943, %944
  %946 = inttoptr i64 %945 to ptr
  %947 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %941, 0
  %948 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %947, ptr %946, 1
  %949 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %948, i64 0, 2
  %950 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %949, i64 1, 3, 0
  %951 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %950, i64 12, 3, 1
  %952 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %951, i64 32, 3, 2
  %953 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %952, i64 1, 3, 3
  %954 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %953, i64 384, 4, 0
  %955 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %954, i64 32, 4, 1
  %956 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %955, i64 1, 4, 2
  %957 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %956, i64 1, 4, 3
  br label %958

958:                                              ; preds = %1066, %923
  %959 = phi i64 [ %1067, %1066 ], [ 0, %923 ]
  %960 = icmp slt i64 %959, 12
  br i1 %960, label %961, label %1068

961:                                              ; preds = %958
  br label %962

962:                                              ; preds = %1050, %961
  %963 = phi i64 [ %1065, %1050 ], [ 0, %961 ]
  %964 = icmp slt i64 %963, 32
  br i1 %964, label %965, label %1066

965:                                              ; preds = %962
  %966 = mul i64 %959, -1
  %967 = add i64 %966, 12
  %968 = call i64 @llvm.smin.i64(i64 %967, i64 8)
  %969 = mul i64 %959, 64
  %970 = mul i64 %963, 2
  %971 = add i64 %969, %970
  %972 = add i64 %971, 1
  %973 = mul i64 %959, 32
  %974 = add i64 %973, %963
  %975 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %931, i64 %974, 2
  %976 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %975, i64 1, 3, 0
  %977 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %976, i64 384, 4, 0
  %978 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %977, i64 %968, 3, 1
  %979 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %978, i64 32, 4, 1
  %980 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %979, i64 8, 3, 2
  %981 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %980, i64 1, 4, 2
  %982 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %981, i64 1, 3, 3
  %983 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %982, i64 1, 4, 3
  %984 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %948, i64 %974, 2
  %985 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %984, i64 1, 3, 0
  %986 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %985, i64 384, 4, 0
  %987 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %986, i64 %968, 3, 1
  %988 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %987, i64 32, 4, 1
  %989 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %988, i64 8, 3, 2
  %990 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %989, i64 1, 4, 2
  %991 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %990, i64 1, 3, 3
  %992 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %991, i64 1, 4, 3
  br label %993

993:                                              ; preds = %1048, %965
  %994 = phi i64 [ %1049, %1048 ], [ 0, %965 ]
  %995 = icmp slt i64 %994, 1
  br i1 %995, label %996, label %1050

996:                                              ; preds = %993
  br label %997

997:                                              ; preds = %1046, %996
  %998 = phi i64 [ %1047, %1046 ], [ 0, %996 ]
  %999 = icmp slt i64 %998, %968
  br i1 %999, label %1000, label %1048

1000:                                             ; preds = %997
  br label %1001

1001:                                             ; preds = %1044, %1000
  %1002 = phi i64 [ %1045, %1044 ], [ 0, %1000 ]
  %1003 = icmp slt i64 %1002, 8
  br i1 %1003, label %1004, label %1046

1004:                                             ; preds = %1001
  br label %1005

1005:                                             ; preds = %1008, %1004
  %1006 = phi i64 [ %1043, %1008 ], [ 0, %1004 ]
  %1007 = icmp slt i64 %1006, 1
  br i1 %1007, label %1008, label %1044

1008:                                             ; preds = %1005
  %1009 = getelementptr float, ptr %380, i64 %971
  %1010 = mul i64 %994, 768
  %1011 = mul i64 %998, 64
  %1012 = add i64 %1010, %1011
  %1013 = mul i64 %1002, 2
  %1014 = add i64 %1012, %1013
  %1015 = add i64 %1014, %1006
  %1016 = getelementptr float, ptr %1009, i64 %1015
  %1017 = load float, ptr %1016, align 4
  %1018 = getelementptr float, ptr %380, i64 %972
  %1019 = getelementptr float, ptr %1018, i64 %1015
  %1020 = load float, ptr %1019, align 4
  %1021 = getelementptr float, ptr %891, i64 %963
  %1022 = add i64 %1002, %1006
  %1023 = getelementptr float, ptr %1021, i64 %1022
  %1024 = load float, ptr %1023, align 4
  %1025 = getelementptr float, ptr %897, i64 %963
  %1026 = getelementptr float, ptr %1025, i64 %1022
  %1027 = load float, ptr %1026, align 4
  %1028 = fmul float %1017, %1024
  %1029 = fmul float %1020, %1027
  %1030 = fsub float %1028, %1029
  %1031 = fmul float %1020, %1024
  %1032 = fmul float %1017, %1027
  %1033 = fadd float %1031, %1032
  %1034 = getelementptr float, ptr %929, i64 %974
  %1035 = mul i64 %994, 384
  %1036 = mul i64 %998, 32
  %1037 = add i64 %1035, %1036
  %1038 = add i64 %1037, %1002
  %1039 = add i64 %1038, %1006
  %1040 = getelementptr float, ptr %1034, i64 %1039
  store float %1030, ptr %1040, align 4
  %1041 = getelementptr float, ptr %946, i64 %974
  %1042 = getelementptr float, ptr %1041, i64 %1039
  store float %1033, ptr %1042, align 4
  %1043 = add i64 %1006, 1
  br label %1005

1044:                                             ; preds = %1005
  %1045 = add i64 %1002, 1
  br label %1001

1046:                                             ; preds = %1001
  %1047 = add i64 %998, 1
  br label %997

1048:                                             ; preds = %997
  %1049 = add i64 %994, 1
  br label %993

1050:                                             ; preds = %993
  %1051 = call ptr @llvm.stacksave.p0()
  %1052 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %983, ptr %1052, align 8
  %1053 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1052, 1
  %1054 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %983, ptr %1054, align 8
  %1055 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1054, 1
  %1056 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1053, ptr %1056, align 8
  %1057 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1055, ptr %1057, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %1056, ptr %1057)
  call void @llvm.stackrestore.p0(ptr %1051)
  %1058 = call ptr @llvm.stacksave.p0()
  %1059 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %992, ptr %1059, align 8
  %1060 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1059, 1
  %1061 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %992, ptr %1061, align 8
  %1062 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1061, 1
  %1063 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1060, ptr %1063, align 8
  %1064 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1062, ptr %1064, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %1063, ptr %1064)
  call void @llvm.stackrestore.p0(ptr %1058)
  %1065 = add i64 %963, 8
  br label %962

1066:                                             ; preds = %962
  %1067 = add i64 %959, 8
  br label %958

1068:                                             ; preds = %958
  %1069 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1070 = ptrtoint ptr %1069 to i64
  %1071 = add i64 %1070, 63
  %1072 = urem i64 %1071, 64
  %1073 = sub i64 %1071, %1072
  %1074 = inttoptr i64 %1073 to ptr
  %1075 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %1069, 0
  %1076 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1075, ptr %1074, 1
  %1077 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1076, i64 0, 2
  %1078 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1077, i64 1, 3, 0
  %1079 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1078, i64 768, 4, 0
  %1080 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1079, i64 12, 3, 1
  %1081 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1080, i64 64, 4, 1
  %1082 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1081, i64 32, 3, 2
  %1083 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1082, i64 2, 4, 2
  %1084 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1083, i64 1, 3, 3
  %1085 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1084, i64 1, 4, 3
  %1086 = call ptr @llvm.stacksave.p0()
  %1087 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %940, ptr %1087, align 8
  %1088 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1087, 1
  %1089 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %1085, ptr %1089, align 8
  %1090 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1089, 1
  %1091 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1088, ptr %1091, align 8
  %1092 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1090, ptr %1092, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %1091, ptr %1092)
  call void @llvm.stackrestore.p0(ptr %1086)
  %1093 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1094 = ptrtoint ptr %1093 to i64
  %1095 = add i64 %1094, 63
  %1096 = urem i64 %1095, 64
  %1097 = sub i64 %1095, %1096
  %1098 = inttoptr i64 %1097 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1098, ptr %1074, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %1099 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %1093, 0
  %1100 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1099, ptr %1098, 1
  %1101 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1100, i64 1, 2
  %1102 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1101, i64 1, 3, 0
  %1103 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1102, i64 768, 4, 0
  %1104 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1103, i64 12, 3, 1
  %1105 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1104, i64 64, 4, 1
  %1106 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1105, i64 32, 3, 2
  %1107 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1106, i64 2, 4, 2
  %1108 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1107, i64 1, 3, 3
  %1109 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1108, i64 1, 4, 3
  %1110 = call ptr @llvm.stacksave.p0()
  %1111 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %957, ptr %1111, align 8
  %1112 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1111, 1
  %1113 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %1109, ptr %1113, align 8
  %1114 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1113, 1
  %1115 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1112, ptr %1115, align 8
  %1116 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1114, ptr %1116, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %1115, ptr %1116)
  call void @llvm.stackrestore.p0(ptr %1110)
  %1117 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %1118 = ptrtoint ptr %1117 to i64
  %1119 = add i64 %1118, 63
  %1120 = urem i64 %1119, 64
  %1121 = sub i64 %1119, %1120
  %1122 = inttoptr i64 %1121 to ptr
  %1123 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, 3, 0
  %1124 = mul i64 %1123, 1
  %1125 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, 3, 1
  %1126 = mul i64 %1124, %1125
  %1127 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, 3, 2
  %1128 = mul i64 %1126, %1127
  %1129 = mul i64 %1128, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %1130 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, 1
  %1131 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, 2
  %1132 = getelementptr float, ptr %1130, i64 %1131
  call void @llvm.memcpy.p0.p0.i64(ptr %1122, ptr %1132, i64 %1129, i1 false)
  %1133 = mul i64 %104, 786432
  %1134 = mul i64 %87, 768
  %1135 = add i64 %1133, %1134
  %1136 = getelementptr float, ptr %1122, i64 %1135
  call void @llvm.memcpy.p0.p0.i64(ptr %1136, ptr %867, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %1137 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %1138 = ptrtoint ptr %1137 to i64
  %1139 = add i64 %1138, 63
  %1140 = urem i64 %1139, 64
  %1141 = sub i64 %1139, %1140
  %1142 = inttoptr i64 %1141 to ptr
  %1143 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %107, 3, 0
  %1144 = mul i64 %1143, 1
  %1145 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %107, 3, 1
  %1146 = mul i64 %1144, %1145
  %1147 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %107, 3, 2
  %1148 = mul i64 %1146, %1147
  %1149 = mul i64 %1148, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %1150 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %107, 1
  %1151 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %107, 2
  %1152 = getelementptr float, ptr %1150, i64 %1151
  call void @llvm.memcpy.p0.p0.i64(ptr %1142, ptr %1152, i64 %1149, i1 false)
  %1153 = getelementptr float, ptr %1142, i64 %1135
  call void @llvm.memcpy.p0.p0.i64(ptr %1153, ptr %1098, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  %1154 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 786432) to i64), i64 64))
  %1155 = ptrtoint ptr %1154 to i64
  %1156 = add i64 %1155, 63
  %1157 = urem i64 %1156, 64
  %1158 = sub i64 %1156, %1157
  %1159 = inttoptr i64 %1158 to ptr
  %1160 = getelementptr float, ptr %1122, i64 %1133
  call void @llvm.memcpy.p0.p0.i64(ptr %1159, ptr %1160, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 786432), i1 false)
  %1161 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 786432) to i64), i64 64))
  %1162 = ptrtoint ptr %1161 to i64
  %1163 = add i64 %1162, 63
  %1164 = urem i64 %1163, 64
  %1165 = sub i64 %1163, %1164
  %1166 = inttoptr i64 %1165 to ptr
  %1167 = getelementptr float, ptr %1142, i64 %1133
  call void @llvm.memcpy.p0.p0.i64(ptr %1166, ptr %1167, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 786432), i1 false)
  %1168 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1169 = ptrtoint ptr %1168 to i64
  %1170 = add i64 %1169, 63
  %1171 = urem i64 %1170, 64
  %1172 = sub i64 %1170, %1171
  %1173 = inttoptr i64 %1172 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1173, ptr @__constant_1x12x64xf32, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %1174

1174:                                             ; preds = %1500, %1068
  %1175 = phi i64 [ %1503, %1500 ], [ 0, %1068 ]
  %1176 = icmp slt i64 %1175, 12
  br i1 %1176, label %1177, label %1504

1177:                                             ; preds = %1174
  %1178 = mul i64 %1175, 64
  %1179 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %1180 = ptrtoint ptr %1179 to i64
  %1181 = add i64 %1180, 63
  %1182 = urem i64 %1181, 64
  %1183 = sub i64 %1181, %1182
  %1184 = inttoptr i64 %1183 to ptr
  br label %1185

1185:                                             ; preds = %1204, %1177
  %1186 = phi i64 [ %1206, %1204 ], [ 0, %1177 ]
  %1187 = icmp slt i64 %1186, 1024
  br i1 %1187, label %1188, label %1207

1188:                                             ; preds = %1185
  br label %1189

1189:                                             ; preds = %1202, %1188
  %1190 = phi i64 [ %1203, %1202 ], [ 0, %1188 ]
  %1191 = icmp slt i64 %1190, 1
  br i1 %1191, label %1192, label %1204

1192:                                             ; preds = %1189
  br label %1193

1193:                                             ; preds = %1196, %1192
  %1194 = phi i64 [ %1201, %1196 ], [ 0, %1192 ]
  %1195 = icmp slt i64 %1194, 8
  br i1 %1195, label %1196, label %1202

1196:                                             ; preds = %1193
  %1197 = getelementptr float, ptr %1184, i64 %1186
  %1198 = mul i64 %1190, 1024
  %1199 = add i64 %1198, %1194
  %1200 = getelementptr float, ptr %1197, i64 %1199
  store float 0.000000e+00, ptr %1200, align 4
  %1201 = add i64 %1194, 1
  br label %1193

1202:                                             ; preds = %1193
  %1203 = add i64 %1190, 1
  br label %1189

1204:                                             ; preds = %1189
  %1205 = getelementptr float, ptr %1184, i64 %1186
  call void @llvm.memcpy.p0.p0.i64(ptr %1205, ptr %1205, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1206 = add i64 %1186, 8
  br label %1185

1207:                                             ; preds = %1185
  br label %1208

1208:                                             ; preds = %1262, %1207
  %1209 = phi i64 [ %1263, %1262 ], [ 0, %1207 ]
  %1210 = icmp slt i64 %1209, 1024
  br i1 %1210, label %1211, label %1264

1211:                                             ; preds = %1208
  br label %1212

1212:                                             ; preds = %1259, %1211
  %1213 = phi i64 [ %1261, %1259 ], [ 0, %1211 ]
  %1214 = icmp slt i64 %1213, 64
  br i1 %1214, label %1215, label %1262

1215:                                             ; preds = %1212
  %1216 = add i64 %1178, %1213
  %1217 = mul i64 %1209, 768
  %1218 = add i64 %1178, %1217
  %1219 = add i64 %1218, %1213
  br label %1220

1220:                                             ; preds = %1257, %1215
  %1221 = phi i64 [ %1258, %1257 ], [ 0, %1215 ]
  %1222 = icmp slt i64 %1221, 1
  br i1 %1222, label %1223, label %1259

1223:                                             ; preds = %1220
  br label %1224

1224:                                             ; preds = %1255, %1223
  %1225 = phi i64 [ %1256, %1255 ], [ 0, %1223 ]
  %1226 = icmp slt i64 %1225, 8
  br i1 %1226, label %1227, label %1257

1227:                                             ; preds = %1224
  br label %1228

1228:                                             ; preds = %1231, %1227
  %1229 = phi i64 [ %1254, %1231 ], [ 0, %1227 ]
  %1230 = icmp slt i64 %1229, 8
  br i1 %1230, label %1231, label %1255

1231:                                             ; preds = %1228
  %1232 = getelementptr float, ptr %636, i64 %1216
  %1233 = mul i64 %1221, 768
  %1234 = add i64 %1233, %1229
  %1235 = getelementptr float, ptr %1232, i64 %1234
  %1236 = load float, ptr %1235, align 4
  %1237 = getelementptr float, ptr %1159, i64 %1219
  %1238 = mul i64 %1225, 768
  %1239 = add i64 %1238, %1229
  %1240 = getelementptr float, ptr %1237, i64 %1239
  %1241 = load float, ptr %1240, align 4
  %1242 = getelementptr float, ptr %1184, i64 %1209
  %1243 = mul i64 %1221, 1024
  %1244 = add i64 %1243, %1225
  %1245 = getelementptr float, ptr %1242, i64 %1244
  %1246 = load float, ptr %1245, align 4
  %1247 = add i64 %1225, %1209
  %1248 = icmp sle i64 %1247, %87
  %1249 = select i1 %1248, float 1.000000e+00, float 0.000000e+00
  %1250 = fmul float %1236, %1241
  %1251 = fadd float %1246, %1250
  %1252 = fcmp ugt float %1249, 5.000000e-01
  %1253 = select i1 %1252, float %1251, float -1.000000e+09
  store float %1253, ptr %1245, align 4
  %1254 = add i64 %1229, 1
  br label %1228

1255:                                             ; preds = %1228
  %1256 = add i64 %1225, 1
  br label %1224

1257:                                             ; preds = %1224
  %1258 = add i64 %1221, 1
  br label %1220

1259:                                             ; preds = %1220
  %1260 = getelementptr float, ptr %1184, i64 %1209
  call void @llvm.memcpy.p0.p0.i64(ptr %1260, ptr %1260, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1261 = add i64 %1213, 8
  br label %1212

1262:                                             ; preds = %1212
  %1263 = add i64 %1209, 8
  br label %1208

1264:                                             ; preds = %1208
  %1265 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %1266 = ptrtoint ptr %1265 to i64
  %1267 = add i64 %1266, 63
  %1268 = urem i64 %1267, 64
  %1269 = sub i64 %1267, %1268
  %1270 = inttoptr i64 %1269 to ptr
  br label %1271

1271:                                             ; preds = %1294, %1264
  %1272 = phi i64 [ %1296, %1294 ], [ 0, %1264 ]
  %1273 = icmp slt i64 %1272, 1024
  br i1 %1273, label %1274, label %1297

1274:                                             ; preds = %1271
  br label %1275

1275:                                             ; preds = %1292, %1274
  %1276 = phi i64 [ %1293, %1292 ], [ 0, %1274 ]
  %1277 = icmp slt i64 %1276, 1
  br i1 %1277, label %1278, label %1294

1278:                                             ; preds = %1275
  br label %1279

1279:                                             ; preds = %1282, %1278
  %1280 = phi i64 [ %1291, %1282 ], [ 0, %1278 ]
  %1281 = icmp slt i64 %1280, 8
  br i1 %1281, label %1282, label %1292

1282:                                             ; preds = %1279
  %1283 = getelementptr float, ptr %1184, i64 %1272
  %1284 = mul i64 %1276, 1024
  %1285 = add i64 %1284, %1280
  %1286 = getelementptr float, ptr %1283, i64 %1285
  %1287 = load float, ptr %1286, align 4
  %1288 = fmul float %1287, 1.250000e-01
  %1289 = getelementptr float, ptr %1270, i64 %1272
  %1290 = getelementptr float, ptr %1289, i64 %1285
  store float %1288, ptr %1290, align 4
  %1291 = add i64 %1280, 1
  br label %1279

1292:                                             ; preds = %1279
  %1293 = add i64 %1276, 1
  br label %1275

1294:                                             ; preds = %1275
  %1295 = getelementptr float, ptr %1270, i64 %1272
  call void @llvm.memcpy.p0.p0.i64(ptr %1295, ptr %1295, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1296 = add i64 %1272, 8
  br label %1271

1297:                                             ; preds = %1271
  %1298 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1299 = ptrtoint ptr %1298 to i64
  %1300 = add i64 %1299, 63
  %1301 = urem i64 %1300, 64
  %1302 = sub i64 %1300, %1301
  %1303 = inttoptr i64 %1302 to ptr
  br label %1304

1304:                                             ; preds = %1307, %1297
  %1305 = phi i64 [ %1309, %1307 ], [ 0, %1297 ]
  %1306 = icmp slt i64 %1305, 1
  br i1 %1306, label %1307, label %1310

1307:                                             ; preds = %1304
  %1308 = getelementptr float, ptr %1303, i64 %1305
  store float 0xFFF0000000000000, ptr %1308, align 4
  %1309 = add i64 %1305, 1
  br label %1304

1310:                                             ; preds = %1304
  br label %1311

1311:                                             ; preds = %1334, %1310
  %1312 = phi i64 [ %1335, %1334 ], [ 0, %1310 ]
  %1313 = icmp slt i64 %1312, 1024
  br i1 %1313, label %1314, label %1336

1314:                                             ; preds = %1311
  br label %1315

1315:                                             ; preds = %1332, %1314
  %1316 = phi i64 [ %1333, %1332 ], [ 0, %1314 ]
  %1317 = icmp slt i64 %1316, 1
  br i1 %1317, label %1318, label %1334

1318:                                             ; preds = %1315
  br label %1319

1319:                                             ; preds = %1322, %1318
  %1320 = phi i64 [ %1331, %1322 ], [ 0, %1318 ]
  %1321 = icmp slt i64 %1320, 8
  br i1 %1321, label %1322, label %1332

1322:                                             ; preds = %1319
  %1323 = getelementptr float, ptr %1270, i64 %1312
  %1324 = mul i64 %1316, 1024
  %1325 = add i64 %1324, %1320
  %1326 = getelementptr float, ptr %1323, i64 %1325
  %1327 = load float, ptr %1326, align 4
  %1328 = getelementptr float, ptr %1303, i64 %1316
  %1329 = load float, ptr %1328, align 4
  %1330 = call float @llvm.maxnum.f32(float %1327, float %1329)
  store float %1330, ptr %1328, align 4
  %1331 = add i64 %1320, 1
  br label %1319

1332:                                             ; preds = %1319
  %1333 = add i64 %1316, 1
  br label %1315

1334:                                             ; preds = %1315
  %1335 = add i64 %1312, 8
  br label %1311

1336:                                             ; preds = %1311
  %1337 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64), i64 64))
  %1338 = ptrtoint ptr %1337 to i64
  %1339 = add i64 %1338, 63
  %1340 = urem i64 %1339, 64
  %1341 = sub i64 %1339, %1340
  %1342 = inttoptr i64 %1341 to ptr
  br label %1343

1343:                                             ; preds = %1369, %1336
  %1344 = phi i64 [ %1371, %1369 ], [ 0, %1336 ]
  %1345 = icmp slt i64 %1344, 1024
  br i1 %1345, label %1346, label %1372

1346:                                             ; preds = %1343
  br label %1347

1347:                                             ; preds = %1367, %1346
  %1348 = phi i64 [ %1368, %1367 ], [ 0, %1346 ]
  %1349 = icmp slt i64 %1348, 1
  br i1 %1349, label %1350, label %1369

1350:                                             ; preds = %1347
  br label %1351

1351:                                             ; preds = %1354, %1350
  %1352 = phi i64 [ %1366, %1354 ], [ 0, %1350 ]
  %1353 = icmp slt i64 %1352, 8
  br i1 %1353, label %1354, label %1367

1354:                                             ; preds = %1351
  %1355 = getelementptr float, ptr %1270, i64 %1344
  %1356 = mul i64 %1348, 1024
  %1357 = add i64 %1356, %1352
  %1358 = getelementptr float, ptr %1355, i64 %1357
  %1359 = load float, ptr %1358, align 4
  %1360 = getelementptr float, ptr %1303, i64 %1348
  %1361 = load float, ptr %1360, align 4
  %1362 = fsub float %1359, %1361
  %1363 = call float @llvm.exp.f32(float %1362)
  %1364 = getelementptr float, ptr %1342, i64 %1344
  %1365 = getelementptr float, ptr %1364, i64 %1357
  store float %1363, ptr %1365, align 4
  %1366 = add i64 %1352, 1
  br label %1351

1367:                                             ; preds = %1351
  %1368 = add i64 %1348, 1
  br label %1347

1369:                                             ; preds = %1347
  %1370 = getelementptr float, ptr %1342, i64 %1344
  call void @llvm.memcpy.p0.p0.i64(ptr %1370, ptr %1370, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1371 = add i64 %1344, 8
  br label %1343

1372:                                             ; preds = %1343
  %1373 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1374 = ptrtoint ptr %1373 to i64
  %1375 = add i64 %1374, 63
  %1376 = urem i64 %1375, 64
  %1377 = sub i64 %1375, %1376
  %1378 = inttoptr i64 %1377 to ptr
  br label %1379

1379:                                             ; preds = %1382, %1372
  %1380 = phi i64 [ %1384, %1382 ], [ 0, %1372 ]
  %1381 = icmp slt i64 %1380, 1
  br i1 %1381, label %1382, label %1385

1382:                                             ; preds = %1379
  %1383 = getelementptr float, ptr %1378, i64 %1380
  store float 0.000000e+00, ptr %1383, align 4
  %1384 = add i64 %1380, 1
  br label %1379

1385:                                             ; preds = %1379
  br label %1386

1386:                                             ; preds = %1409, %1385
  %1387 = phi i64 [ %1410, %1409 ], [ 0, %1385 ]
  %1388 = icmp slt i64 %1387, 1024
  br i1 %1388, label %1389, label %1411

1389:                                             ; preds = %1386
  br label %1390

1390:                                             ; preds = %1407, %1389
  %1391 = phi i64 [ %1408, %1407 ], [ 0, %1389 ]
  %1392 = icmp slt i64 %1391, 1
  br i1 %1392, label %1393, label %1409

1393:                                             ; preds = %1390
  br label %1394

1394:                                             ; preds = %1397, %1393
  %1395 = phi i64 [ %1406, %1397 ], [ 0, %1393 ]
  %1396 = icmp slt i64 %1395, 8
  br i1 %1396, label %1397, label %1407

1397:                                             ; preds = %1394
  %1398 = getelementptr float, ptr %1342, i64 %1387
  %1399 = mul i64 %1391, 1024
  %1400 = add i64 %1399, %1395
  %1401 = getelementptr float, ptr %1398, i64 %1400
  %1402 = load float, ptr %1401, align 4
  %1403 = getelementptr float, ptr %1378, i64 %1391
  %1404 = load float, ptr %1403, align 4
  %1405 = fadd float %1402, %1404
  store float %1405, ptr %1403, align 4
  %1406 = add i64 %1395, 1
  br label %1394

1407:                                             ; preds = %1394
  %1408 = add i64 %1391, 1
  br label %1390

1409:                                             ; preds = %1390
  %1410 = add i64 %1387, 8
  br label %1386

1411:                                             ; preds = %1386
  %1412 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64), i64 64))
  %1413 = ptrtoint ptr %1412 to i64
  %1414 = add i64 %1413, 63
  %1415 = urem i64 %1414, 64
  %1416 = sub i64 %1414, %1415
  %1417 = inttoptr i64 %1416 to ptr
  br label %1418

1418:                                             ; preds = %1437, %1411
  %1419 = phi i64 [ %1439, %1437 ], [ 0, %1411 ]
  %1420 = icmp slt i64 %1419, 64
  br i1 %1420, label %1421, label %1440

1421:                                             ; preds = %1418
  br label %1422

1422:                                             ; preds = %1435, %1421
  %1423 = phi i64 [ %1436, %1435 ], [ 0, %1421 ]
  %1424 = icmp slt i64 %1423, 1
  br i1 %1424, label %1425, label %1437

1425:                                             ; preds = %1422
  br label %1426

1426:                                             ; preds = %1429, %1425
  %1427 = phi i64 [ %1434, %1429 ], [ 0, %1425 ]
  %1428 = icmp slt i64 %1427, 8
  br i1 %1428, label %1429, label %1435

1429:                                             ; preds = %1426
  %1430 = getelementptr float, ptr %1417, i64 %1419
  %1431 = mul i64 %1423, 64
  %1432 = add i64 %1431, %1427
  %1433 = getelementptr float, ptr %1430, i64 %1432
  store float 0.000000e+00, ptr %1433, align 4
  %1434 = add i64 %1427, 1
  br label %1426

1435:                                             ; preds = %1426
  %1436 = add i64 %1423, 1
  br label %1422

1437:                                             ; preds = %1422
  %1438 = getelementptr float, ptr %1417, i64 %1419
  call void @llvm.memcpy.p0.p0.i64(ptr %1438, ptr %1438, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1439 = add i64 %1419, 8
  br label %1418

1440:                                             ; preds = %1418
  %1441 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64), i64 64))
  %1442 = ptrtoint ptr %1441 to i64
  %1443 = add i64 %1442, 63
  %1444 = urem i64 %1443, 64
  %1445 = sub i64 %1443, %1444
  %1446 = inttoptr i64 %1445 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1446, ptr %1417, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64), i1 false)
  br label %1447

1447:                                             ; preds = %1498, %1440
  %1448 = phi i64 [ %1499, %1498 ], [ 0, %1440 ]
  %1449 = icmp slt i64 %1448, 64
  br i1 %1449, label %1450, label %1500

1450:                                             ; preds = %1447
  br label %1451

1451:                                             ; preds = %1495, %1450
  %1452 = phi i64 [ %1497, %1495 ], [ 0, %1450 ]
  %1453 = icmp slt i64 %1452, 1024
  br i1 %1453, label %1454, label %1498

1454:                                             ; preds = %1451
  %1455 = mul i64 %1452, 768
  %1456 = add i64 %1178, %1455
  %1457 = add i64 %1456, %1448
  br label %1458

1458:                                             ; preds = %1493, %1454
  %1459 = phi i64 [ %1494, %1493 ], [ 0, %1454 ]
  %1460 = icmp slt i64 %1459, 1
  br i1 %1460, label %1461, label %1495

1461:                                             ; preds = %1458
  br label %1462

1462:                                             ; preds = %1491, %1461
  %1463 = phi i64 [ %1492, %1491 ], [ 0, %1461 ]
  %1464 = icmp slt i64 %1463, 8
  br i1 %1464, label %1465, label %1493

1465:                                             ; preds = %1462
  br label %1466

1466:                                             ; preds = %1469, %1465
  %1467 = phi i64 [ %1490, %1469 ], [ 0, %1465 ]
  %1468 = icmp slt i64 %1467, 8
  br i1 %1468, label %1469, label %1491

1469:                                             ; preds = %1466
  %1470 = getelementptr float, ptr %1342, i64 %1452
  %1471 = mul i64 %1459, 1024
  %1472 = add i64 %1471, %1467
  %1473 = getelementptr float, ptr %1470, i64 %1472
  %1474 = load float, ptr %1473, align 4
  %1475 = getelementptr float, ptr %1378, i64 %1459
  %1476 = load float, ptr %1475, align 4
  %1477 = getelementptr float, ptr %1166, i64 %1457
  %1478 = mul i64 %1467, 768
  %1479 = add i64 %1478, %1463
  %1480 = getelementptr float, ptr %1477, i64 %1479
  %1481 = load float, ptr %1480, align 4
  %1482 = getelementptr float, ptr %1446, i64 %1448
  %1483 = mul i64 %1459, 64
  %1484 = add i64 %1483, %1463
  %1485 = getelementptr float, ptr %1482, i64 %1484
  %1486 = load float, ptr %1485, align 4
  %1487 = fdiv float %1474, %1476
  %1488 = fmul float %1487, %1481
  %1489 = fadd float %1486, %1488
  store float %1489, ptr %1485, align 4
  %1490 = add i64 %1467, 1
  br label %1466

1491:                                             ; preds = %1466
  %1492 = add i64 %1463, 1
  br label %1462

1493:                                             ; preds = %1462
  %1494 = add i64 %1459, 1
  br label %1458

1495:                                             ; preds = %1458
  %1496 = getelementptr float, ptr %1446, i64 %1448
  call void @llvm.memcpy.p0.p0.i64(ptr %1496, ptr %1496, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1497 = add i64 %1452, 8
  br label %1451

1498:                                             ; preds = %1451
  %1499 = add i64 %1448, 8
  br label %1447

1500:                                             ; preds = %1447
  %1501 = mul i64 %1175, 64
  %1502 = getelementptr float, ptr %1173, i64 %1501
  call void @llvm.memcpy.p0.p0.i64(ptr %1502, ptr %1446, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64), i1 false)
  %1503 = add i64 %1175, 1
  br label %1174

1504:                                             ; preds = %1174
  %1505 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1506 = ptrtoint ptr %1505 to i64
  %1507 = add i64 %1506, 63
  %1508 = urem i64 %1507, 64
  %1509 = sub i64 %1507, %1508
  %1510 = inttoptr i64 %1509 to ptr
  br label %1511

1511:                                             ; preds = %1530, %1504
  %1512 = phi i64 [ %1532, %1530 ], [ 0, %1504 ]
  %1513 = icmp slt i64 %1512, 768
  br i1 %1513, label %1514, label %1533

1514:                                             ; preds = %1511
  br label %1515

1515:                                             ; preds = %1528, %1514
  %1516 = phi i64 [ %1529, %1528 ], [ 0, %1514 ]
  %1517 = icmp slt i64 %1516, 1
  br i1 %1517, label %1518, label %1530

1518:                                             ; preds = %1515
  br label %1519

1519:                                             ; preds = %1522, %1518
  %1520 = phi i64 [ %1527, %1522 ], [ 0, %1518 ]
  %1521 = icmp slt i64 %1520, 8
  br i1 %1521, label %1522, label %1528

1522:                                             ; preds = %1519
  %1523 = getelementptr float, ptr %1510, i64 %1512
  %1524 = mul i64 %1516, 768
  %1525 = add i64 %1524, %1520
  %1526 = getelementptr float, ptr %1523, i64 %1525
  store float 0.000000e+00, ptr %1526, align 4
  %1527 = add i64 %1520, 1
  br label %1519

1528:                                             ; preds = %1519
  %1529 = add i64 %1516, 1
  br label %1515

1530:                                             ; preds = %1515
  %1531 = getelementptr float, ptr %1510, i64 %1512
  call void @llvm.memcpy.p0.p0.i64(ptr %1531, ptr %1531, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1532 = add i64 %1512, 8
  br label %1511

1533:                                             ; preds = %1511
  br label %1534

1534:                                             ; preds = %1573, %1533
  %1535 = phi i64 [ %1574, %1573 ], [ 0, %1533 ]
  %1536 = icmp slt i64 %1535, 768
  br i1 %1536, label %1537, label %1575

1537:                                             ; preds = %1534
  br label %1538

1538:                                             ; preds = %1570, %1537
  %1539 = phi i64 [ %1572, %1570 ], [ 0, %1537 ]
  %1540 = icmp slt i64 %1539, 768
  br i1 %1540, label %1541, label %1573

1541:                                             ; preds = %1538
  br label %1542

1542:                                             ; preds = %1568, %1541
  %1543 = phi i64 [ %1569, %1568 ], [ 0, %1541 ]
  %1544 = icmp slt i64 %1543, 1
  br i1 %1544, label %1545, label %1570

1545:                                             ; preds = %1542
  br label %1546

1546:                                             ; preds = %1566, %1545
  %1547 = phi i64 [ %1567, %1566 ], [ 0, %1545 ]
  %1548 = icmp slt i64 %1547, 8
  br i1 %1548, label %1549, label %1568

1549:                                             ; preds = %1546
  br label %1550

1550:                                             ; preds = %1553, %1549
  %1551 = phi i64 [ %1565, %1553 ], [ 0, %1549 ]
  %1552 = icmp slt i64 %1551, 8
  br i1 %1552, label %1553, label %1566

1553:                                             ; preds = %1550
  %1554 = getelementptr float, ptr %1173, i64 %1539
  %1555 = mul i64 %1543, 768
  %1556 = add i64 %1555, %1551
  %1557 = getelementptr float, ptr %1554, i64 %1556
  %1558 = load float, ptr %1557, align 4
  %1559 = getelementptr float, ptr %1510, i64 %1535
  %1560 = add i64 %1555, %1547
  %1561 = getelementptr float, ptr %1559, i64 %1560
  %1562 = load float, ptr %1561, align 4
  %1563 = fmul float %1558, 7.000000e+00
  %1564 = fadd float %1562, %1563
  store float %1564, ptr %1561, align 4
  %1565 = add i64 %1551, 1
  br label %1550

1566:                                             ; preds = %1550
  %1567 = add i64 %1547, 1
  br label %1546

1568:                                             ; preds = %1546
  %1569 = add i64 %1543, 1
  br label %1542

1570:                                             ; preds = %1542
  %1571 = getelementptr float, ptr %1510, i64 %1535
  call void @llvm.memcpy.p0.p0.i64(ptr %1571, ptr %1571, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1572 = add i64 %1539, 8
  br label %1538

1573:                                             ; preds = %1538
  %1574 = add i64 %1535, 8
  br label %1534

1575:                                             ; preds = %1534
  %1576 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1577 = ptrtoint ptr %1576 to i64
  %1578 = add i64 %1577, 63
  %1579 = urem i64 %1578, 64
  %1580 = sub i64 %1578, %1579
  %1581 = inttoptr i64 %1580 to ptr
  br label %1582

1582:                                             ; preds = %1609, %1575
  %1583 = phi i64 [ %1611, %1609 ], [ 0, %1575 ]
  %1584 = icmp slt i64 %1583, 768
  br i1 %1584, label %1585, label %1612

1585:                                             ; preds = %1582
  %1586 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %105, 1
  br label %1587

1587:                                             ; preds = %1607, %1585
  %1588 = phi i64 [ %1608, %1607 ], [ 0, %1585 ]
  %1589 = icmp slt i64 %1588, 1
  br i1 %1589, label %1590, label %1609

1590:                                             ; preds = %1587
  br label %1591

1591:                                             ; preds = %1594, %1590
  %1592 = phi i64 [ %1606, %1594 ], [ 0, %1590 ]
  %1593 = icmp slt i64 %1592, 8
  br i1 %1593, label %1594, label %1607

1594:                                             ; preds = %1591
  %1595 = getelementptr float, ptr %1586, i64 %1583
  %1596 = mul i64 %1588, 768
  %1597 = add i64 %1596, %1592
  %1598 = getelementptr float, ptr %1595, i64 %1597
  %1599 = load float, ptr %1598, align 4
  %1600 = getelementptr float, ptr %1510, i64 %1583
  %1601 = getelementptr float, ptr %1600, i64 %1597
  %1602 = load float, ptr %1601, align 4
  %1603 = fadd float %1599, %1602
  %1604 = getelementptr float, ptr %1581, i64 %1583
  %1605 = getelementptr float, ptr %1604, i64 %1597
  store float %1603, ptr %1605, align 4
  %1606 = add i64 %1592, 1
  br label %1591

1607:                                             ; preds = %1591
  %1608 = add i64 %1588, 1
  br label %1587

1609:                                             ; preds = %1587
  %1610 = getelementptr float, ptr %1581, i64 %1583
  call void @llvm.memcpy.p0.p0.i64(ptr %1610, ptr %1610, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1611 = add i64 %1583, 8
  br label %1582

1612:                                             ; preds = %1582
  %1613 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1614 = ptrtoint ptr %1613 to i64
  %1615 = add i64 %1614, 63
  %1616 = urem i64 %1615, 64
  %1617 = sub i64 %1615, %1616
  %1618 = inttoptr i64 %1617 to ptr
  br label %1619

1619:                                             ; preds = %1622, %1612
  %1620 = phi i64 [ %1624, %1622 ], [ 0, %1612 ]
  %1621 = icmp slt i64 %1620, 1
  br i1 %1621, label %1622, label %1625

1622:                                             ; preds = %1619
  %1623 = getelementptr float, ptr %1618, i64 %1620
  store float 0.000000e+00, ptr %1623, align 4
  %1624 = add i64 %1620, 1
  br label %1619

1625:                                             ; preds = %1619
  br label %1626

1626:                                             ; preds = %1650, %1625
  %1627 = phi i64 [ %1651, %1650 ], [ 0, %1625 ]
  %1628 = icmp slt i64 %1627, 768
  br i1 %1628, label %1629, label %1652

1629:                                             ; preds = %1626
  br label %1630

1630:                                             ; preds = %1648, %1629
  %1631 = phi i64 [ %1649, %1648 ], [ 0, %1629 ]
  %1632 = icmp slt i64 %1631, 1
  br i1 %1632, label %1633, label %1650

1633:                                             ; preds = %1630
  br label %1634

1634:                                             ; preds = %1637, %1633
  %1635 = phi i64 [ %1647, %1637 ], [ 0, %1633 ]
  %1636 = icmp slt i64 %1635, 8
  br i1 %1636, label %1637, label %1648

1637:                                             ; preds = %1634
  %1638 = getelementptr float, ptr %1581, i64 %1627
  %1639 = mul i64 %1631, 768
  %1640 = add i64 %1639, %1635
  %1641 = getelementptr float, ptr %1638, i64 %1640
  %1642 = load float, ptr %1641, align 4
  %1643 = getelementptr float, ptr %1618, i64 %1631
  %1644 = load float, ptr %1643, align 4
  %1645 = fmul float %1642, %1642
  %1646 = fadd float %1644, %1645
  store float %1646, ptr %1643, align 4
  %1647 = add i64 %1635, 1
  br label %1634

1648:                                             ; preds = %1634
  %1649 = add i64 %1631, 1
  br label %1630

1650:                                             ; preds = %1630
  %1651 = add i64 %1627, 8
  br label %1626

1652:                                             ; preds = %1626
  %1653 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1654 = ptrtoint ptr %1653 to i64
  %1655 = add i64 %1654, 63
  %1656 = urem i64 %1655, 64
  %1657 = sub i64 %1655, %1656
  %1658 = inttoptr i64 %1657 to ptr
  br label %1659

1659:                                             ; preds = %1689, %1652
  %1660 = phi i64 [ %1691, %1689 ], [ 0, %1652 ]
  %1661 = icmp slt i64 %1660, 768
  br i1 %1661, label %1662, label %1692

1662:                                             ; preds = %1659
  br label %1663

1663:                                             ; preds = %1687, %1662
  %1664 = phi i64 [ %1688, %1687 ], [ 0, %1662 ]
  %1665 = icmp slt i64 %1664, 1
  br i1 %1665, label %1666, label %1689

1666:                                             ; preds = %1663
  br label %1667

1667:                                             ; preds = %1670, %1666
  %1668 = phi i64 [ %1686, %1670 ], [ 0, %1666 ]
  %1669 = icmp slt i64 %1668, 8
  br i1 %1669, label %1670, label %1687

1670:                                             ; preds = %1667
  %1671 = getelementptr float, ptr %1581, i64 %1660
  %1672 = mul i64 %1664, 768
  %1673 = add i64 %1672, %1668
  %1674 = getelementptr float, ptr %1671, i64 %1673
  %1675 = load float, ptr %1674, align 4
  %1676 = getelementptr float, ptr %1618, i64 %1664
  %1677 = load float, ptr %1676, align 4
  %1678 = fdiv float %1677, 7.680000e+02
  %1679 = fadd float %1678, 0x3EE4F8B580000000
  %1680 = call float @llvm.sqrt.f32(float %1679)
  %1681 = fdiv float 1.000000e+00, %1680
  %1682 = fmul float %1675, %1681
  %1683 = fmul float %1682, 8.000000e+00
  %1684 = getelementptr float, ptr %1658, i64 %1660
  %1685 = getelementptr float, ptr %1684, i64 %1673
  store float %1683, ptr %1685, align 4
  %1686 = add i64 %1668, 1
  br label %1667

1687:                                             ; preds = %1667
  %1688 = add i64 %1664, 1
  br label %1663

1689:                                             ; preds = %1663
  %1690 = getelementptr float, ptr %1658, i64 %1660
  call void @llvm.memcpy.p0.p0.i64(ptr %1690, ptr %1690, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1691 = add i64 %1660, 8
  br label %1659

1692:                                             ; preds = %1659
  %1693 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %1694 = ptrtoint ptr %1693 to i64
  %1695 = add i64 %1694, 63
  %1696 = urem i64 %1695, 64
  %1697 = sub i64 %1695, %1696
  %1698 = inttoptr i64 %1697 to ptr
  br label %1699

1699:                                             ; preds = %1718, %1692
  %1700 = phi i64 [ %1720, %1718 ], [ 0, %1692 ]
  %1701 = icmp slt i64 %1700, 2048
  br i1 %1701, label %1702, label %1721

1702:                                             ; preds = %1699
  br label %1703

1703:                                             ; preds = %1716, %1702
  %1704 = phi i64 [ %1717, %1716 ], [ 0, %1702 ]
  %1705 = icmp slt i64 %1704, 1
  br i1 %1705, label %1706, label %1718

1706:                                             ; preds = %1703
  br label %1707

1707:                                             ; preds = %1710, %1706
  %1708 = phi i64 [ %1715, %1710 ], [ 0, %1706 ]
  %1709 = icmp slt i64 %1708, 8
  br i1 %1709, label %1710, label %1716

1710:                                             ; preds = %1707
  %1711 = getelementptr float, ptr %1698, i64 %1700
  %1712 = mul i64 %1704, 2048
  %1713 = add i64 %1712, %1708
  %1714 = getelementptr float, ptr %1711, i64 %1713
  store float 0.000000e+00, ptr %1714, align 4
  %1715 = add i64 %1708, 1
  br label %1707

1716:                                             ; preds = %1707
  %1717 = add i64 %1704, 1
  br label %1703

1718:                                             ; preds = %1703
  %1719 = getelementptr float, ptr %1698, i64 %1700
  call void @llvm.memcpy.p0.p0.i64(ptr %1719, ptr %1719, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1720 = add i64 %1700, 8
  br label %1699

1721:                                             ; preds = %1699
  br label %1722

1722:                                             ; preds = %1762, %1721
  %1723 = phi i64 [ %1763, %1762 ], [ 0, %1721 ]
  %1724 = icmp slt i64 %1723, 2048
  br i1 %1724, label %1725, label %1764

1725:                                             ; preds = %1722
  br label %1726

1726:                                             ; preds = %1759, %1725
  %1727 = phi i64 [ %1761, %1759 ], [ 0, %1725 ]
  %1728 = icmp slt i64 %1727, 768
  br i1 %1728, label %1729, label %1762

1729:                                             ; preds = %1726
  br label %1730

1730:                                             ; preds = %1757, %1729
  %1731 = phi i64 [ %1758, %1757 ], [ 0, %1729 ]
  %1732 = icmp slt i64 %1731, 1
  br i1 %1732, label %1733, label %1759

1733:                                             ; preds = %1730
  br label %1734

1734:                                             ; preds = %1755, %1733
  %1735 = phi i64 [ %1756, %1755 ], [ 0, %1733 ]
  %1736 = icmp slt i64 %1735, 8
  br i1 %1736, label %1737, label %1757

1737:                                             ; preds = %1734
  br label %1738

1738:                                             ; preds = %1741, %1737
  %1739 = phi i64 [ %1754, %1741 ], [ 0, %1737 ]
  %1740 = icmp slt i64 %1739, 8
  br i1 %1740, label %1741, label %1755

1741:                                             ; preds = %1738
  %1742 = getelementptr float, ptr %1658, i64 %1727
  %1743 = mul i64 %1731, 768
  %1744 = add i64 %1743, %1739
  %1745 = getelementptr float, ptr %1742, i64 %1744
  %1746 = load float, ptr %1745, align 4
  %1747 = getelementptr float, ptr %1698, i64 %1723
  %1748 = mul i64 %1731, 2048
  %1749 = add i64 %1748, %1735
  %1750 = getelementptr float, ptr %1747, i64 %1749
  %1751 = load float, ptr %1750, align 4
  %1752 = fmul float %1746, 9.000000e+00
  %1753 = fadd float %1751, %1752
  store float %1753, ptr %1750, align 4
  %1754 = add i64 %1739, 1
  br label %1738

1755:                                             ; preds = %1738
  %1756 = add i64 %1735, 1
  br label %1734

1757:                                             ; preds = %1734
  %1758 = add i64 %1731, 1
  br label %1730

1759:                                             ; preds = %1730
  %1760 = getelementptr float, ptr %1698, i64 %1723
  call void @llvm.memcpy.p0.p0.i64(ptr %1760, ptr %1760, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1761 = add i64 %1727, 8
  br label %1726

1762:                                             ; preds = %1726
  %1763 = add i64 %1723, 8
  br label %1722

1764:                                             ; preds = %1722
  %1765 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %1766 = ptrtoint ptr %1765 to i64
  %1767 = add i64 %1766, 63
  %1768 = urem i64 %1767, 64
  %1769 = sub i64 %1767, %1768
  %1770 = inttoptr i64 %1769 to ptr
  br label %1771

1771:                                             ; preds = %1790, %1764
  %1772 = phi i64 [ %1792, %1790 ], [ 0, %1764 ]
  %1773 = icmp slt i64 %1772, 2048
  br i1 %1773, label %1774, label %1793

1774:                                             ; preds = %1771
  br label %1775

1775:                                             ; preds = %1788, %1774
  %1776 = phi i64 [ %1789, %1788 ], [ 0, %1774 ]
  %1777 = icmp slt i64 %1776, 1
  br i1 %1777, label %1778, label %1790

1778:                                             ; preds = %1775
  br label %1779

1779:                                             ; preds = %1782, %1778
  %1780 = phi i64 [ %1787, %1782 ], [ 0, %1778 ]
  %1781 = icmp slt i64 %1780, 8
  br i1 %1781, label %1782, label %1788

1782:                                             ; preds = %1779
  %1783 = getelementptr float, ptr %1770, i64 %1772
  %1784 = mul i64 %1776, 2048
  %1785 = add i64 %1784, %1780
  %1786 = getelementptr float, ptr %1783, i64 %1785
  store float 0.000000e+00, ptr %1786, align 4
  %1787 = add i64 %1780, 1
  br label %1779

1788:                                             ; preds = %1779
  %1789 = add i64 %1776, 1
  br label %1775

1790:                                             ; preds = %1775
  %1791 = getelementptr float, ptr %1770, i64 %1772
  call void @llvm.memcpy.p0.p0.i64(ptr %1791, ptr %1791, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1792 = add i64 %1772, 8
  br label %1771

1793:                                             ; preds = %1771
  br label %1794

1794:                                             ; preds = %1834, %1793
  %1795 = phi i64 [ %1835, %1834 ], [ 0, %1793 ]
  %1796 = icmp slt i64 %1795, 2048
  br i1 %1796, label %1797, label %1836

1797:                                             ; preds = %1794
  br label %1798

1798:                                             ; preds = %1831, %1797
  %1799 = phi i64 [ %1833, %1831 ], [ 0, %1797 ]
  %1800 = icmp slt i64 %1799, 768
  br i1 %1800, label %1801, label %1834

1801:                                             ; preds = %1798
  br label %1802

1802:                                             ; preds = %1829, %1801
  %1803 = phi i64 [ %1830, %1829 ], [ 0, %1801 ]
  %1804 = icmp slt i64 %1803, 1
  br i1 %1804, label %1805, label %1831

1805:                                             ; preds = %1802
  br label %1806

1806:                                             ; preds = %1827, %1805
  %1807 = phi i64 [ %1828, %1827 ], [ 0, %1805 ]
  %1808 = icmp slt i64 %1807, 8
  br i1 %1808, label %1809, label %1829

1809:                                             ; preds = %1806
  br label %1810

1810:                                             ; preds = %1813, %1809
  %1811 = phi i64 [ %1826, %1813 ], [ 0, %1809 ]
  %1812 = icmp slt i64 %1811, 8
  br i1 %1812, label %1813, label %1827

1813:                                             ; preds = %1810
  %1814 = getelementptr float, ptr %1658, i64 %1799
  %1815 = mul i64 %1803, 768
  %1816 = add i64 %1815, %1811
  %1817 = getelementptr float, ptr %1814, i64 %1816
  %1818 = load float, ptr %1817, align 4
  %1819 = getelementptr float, ptr %1770, i64 %1795
  %1820 = mul i64 %1803, 2048
  %1821 = add i64 %1820, %1807
  %1822 = getelementptr float, ptr %1819, i64 %1821
  %1823 = load float, ptr %1822, align 4
  %1824 = fmul float %1818, 1.100000e+01
  %1825 = fadd float %1823, %1824
  store float %1825, ptr %1822, align 4
  %1826 = add i64 %1811, 1
  br label %1810

1827:                                             ; preds = %1810
  %1828 = add i64 %1807, 1
  br label %1806

1829:                                             ; preds = %1806
  %1830 = add i64 %1803, 1
  br label %1802

1831:                                             ; preds = %1802
  %1832 = getelementptr float, ptr %1770, i64 %1795
  call void @llvm.memcpy.p0.p0.i64(ptr %1832, ptr %1832, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1833 = add i64 %1799, 8
  br label %1798

1834:                                             ; preds = %1798
  %1835 = add i64 %1795, 8
  br label %1794

1836:                                             ; preds = %1794
  %1837 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1838 = ptrtoint ptr %1837 to i64
  %1839 = add i64 %1838, 63
  %1840 = urem i64 %1839, 64
  %1841 = sub i64 %1839, %1840
  %1842 = inttoptr i64 %1841 to ptr
  br label %1843

1843:                                             ; preds = %1862, %1836
  %1844 = phi i64 [ %1864, %1862 ], [ 0, %1836 ]
  %1845 = icmp slt i64 %1844, 768
  br i1 %1845, label %1846, label %1865

1846:                                             ; preds = %1843
  br label %1847

1847:                                             ; preds = %1860, %1846
  %1848 = phi i64 [ %1861, %1860 ], [ 0, %1846 ]
  %1849 = icmp slt i64 %1848, 1
  br i1 %1849, label %1850, label %1862

1850:                                             ; preds = %1847
  br label %1851

1851:                                             ; preds = %1854, %1850
  %1852 = phi i64 [ %1859, %1854 ], [ 0, %1850 ]
  %1853 = icmp slt i64 %1852, 8
  br i1 %1853, label %1854, label %1860

1854:                                             ; preds = %1851
  %1855 = getelementptr float, ptr %1842, i64 %1844
  %1856 = mul i64 %1848, 768
  %1857 = add i64 %1856, %1852
  %1858 = getelementptr float, ptr %1855, i64 %1857
  store float 0.000000e+00, ptr %1858, align 4
  %1859 = add i64 %1852, 1
  br label %1851

1860:                                             ; preds = %1851
  %1861 = add i64 %1848, 1
  br label %1847

1862:                                             ; preds = %1847
  %1863 = getelementptr float, ptr %1842, i64 %1844
  call void @llvm.memcpy.p0.p0.i64(ptr %1863, ptr %1863, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1864 = add i64 %1844, 8
  br label %1843

1865:                                             ; preds = %1843
  br label %1866

1866:                                             ; preds = %1914, %1865
  %1867 = phi i64 [ %1915, %1914 ], [ 0, %1865 ]
  %1868 = icmp slt i64 %1867, 768
  br i1 %1868, label %1869, label %1916

1869:                                             ; preds = %1866
  br label %1870

1870:                                             ; preds = %1911, %1869
  %1871 = phi i64 [ %1913, %1911 ], [ 0, %1869 ]
  %1872 = icmp slt i64 %1871, 2048
  br i1 %1872, label %1873, label %1914

1873:                                             ; preds = %1870
  br label %1874

1874:                                             ; preds = %1909, %1873
  %1875 = phi i64 [ %1910, %1909 ], [ 0, %1873 ]
  %1876 = icmp slt i64 %1875, 1
  br i1 %1876, label %1877, label %1911

1877:                                             ; preds = %1874
  br label %1878

1878:                                             ; preds = %1907, %1877
  %1879 = phi i64 [ %1908, %1907 ], [ 0, %1877 ]
  %1880 = icmp slt i64 %1879, 8
  br i1 %1880, label %1881, label %1909

1881:                                             ; preds = %1878
  br label %1882

1882:                                             ; preds = %1885, %1881
  %1883 = phi i64 [ %1906, %1885 ], [ 0, %1881 ]
  %1884 = icmp slt i64 %1883, 8
  br i1 %1884, label %1885, label %1907

1885:                                             ; preds = %1882
  %1886 = getelementptr float, ptr %1698, i64 %1871
  %1887 = mul i64 %1875, 2048
  %1888 = add i64 %1887, %1883
  %1889 = getelementptr float, ptr %1886, i64 %1888
  %1890 = load float, ptr %1889, align 4
  %1891 = getelementptr float, ptr %1770, i64 %1871
  %1892 = getelementptr float, ptr %1891, i64 %1888
  %1893 = load float, ptr %1892, align 4
  %1894 = getelementptr float, ptr %1842, i64 %1867
  %1895 = mul i64 %1875, 768
  %1896 = add i64 %1895, %1879
  %1897 = getelementptr float, ptr %1894, i64 %1896
  %1898 = load float, ptr %1897, align 4
  %1899 = fneg float %1890
  %1900 = call float @llvm.exp.f32(float %1899)
  %1901 = fadd float %1890, %1900
  %1902 = fdiv float %1890, %1901
  %1903 = fmul float %1902, %1893
  %1904 = fmul float %1903, 1.000000e+01
  %1905 = fadd float %1898, %1904
  store float %1905, ptr %1897, align 4
  %1906 = add i64 %1883, 1
  br label %1882

1907:                                             ; preds = %1882
  %1908 = add i64 %1879, 1
  br label %1878

1909:                                             ; preds = %1878
  %1910 = add i64 %1875, 1
  br label %1874

1911:                                             ; preds = %1874
  %1912 = getelementptr float, ptr %1842, i64 %1867
  call void @llvm.memcpy.p0.p0.i64(ptr %1912, ptr %1912, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1913 = add i64 %1871, 8
  br label %1870

1914:                                             ; preds = %1870
  %1915 = add i64 %1867, 8
  br label %1866

1916:                                             ; preds = %1866
  %1917 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %1918 = ptrtoint ptr %1917 to i64
  %1919 = add i64 %1918, 63
  %1920 = urem i64 %1919, 64
  %1921 = sub i64 %1919, %1920
  %1922 = inttoptr i64 %1921 to ptr
  %1923 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1917, 0
  %1924 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1923, ptr %1922, 1
  %1925 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1924, i64 0, 2
  %1926 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1925, i64 1, 3, 0
  %1927 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1926, i64 768, 3, 1
  %1928 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1927, i64 768, 4, 0
  %1929 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1928, i64 1, 4, 1
  br label %1930

1930:                                             ; preds = %1956, %1916
  %1931 = phi i64 [ %1958, %1956 ], [ 0, %1916 ]
  %1932 = icmp slt i64 %1931, 768
  br i1 %1932, label %1933, label %1959

1933:                                             ; preds = %1930
  br label %1934

1934:                                             ; preds = %1954, %1933
  %1935 = phi i64 [ %1955, %1954 ], [ 0, %1933 ]
  %1936 = icmp slt i64 %1935, 1
  br i1 %1936, label %1937, label %1956

1937:                                             ; preds = %1934
  br label %1938

1938:                                             ; preds = %1941, %1937
  %1939 = phi i64 [ %1953, %1941 ], [ 0, %1937 ]
  %1940 = icmp slt i64 %1939, 8
  br i1 %1940, label %1941, label %1954

1941:                                             ; preds = %1938
  %1942 = getelementptr float, ptr %1581, i64 %1931
  %1943 = mul i64 %1935, 768
  %1944 = add i64 %1943, %1939
  %1945 = getelementptr float, ptr %1942, i64 %1944
  %1946 = load float, ptr %1945, align 4
  %1947 = getelementptr float, ptr %1842, i64 %1931
  %1948 = getelementptr float, ptr %1947, i64 %1944
  %1949 = load float, ptr %1948, align 4
  %1950 = fadd float %1946, %1949
  %1951 = getelementptr float, ptr %1922, i64 %1931
  %1952 = getelementptr float, ptr %1951, i64 %1944
  store float %1950, ptr %1952, align 4
  %1953 = add i64 %1939, 1
  br label %1938

1954:                                             ; preds = %1938
  %1955 = add i64 %1935, 1
  br label %1934

1956:                                             ; preds = %1934
  %1957 = getelementptr float, ptr %1922, i64 %1931
  call void @llvm.memcpy.p0.p0.i64(ptr %1957, ptr %1957, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %1958 = add i64 %1931, 8
  br label %1930

1959:                                             ; preds = %1930
  %1960 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %1961 = ptrtoint ptr %1960 to i64
  %1962 = add i64 %1961, 63
  %1963 = urem i64 %1962, 64
  %1964 = sub i64 %1962, %1963
  %1965 = inttoptr i64 %1964 to ptr
  %1966 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1960, 0
  %1967 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1966, ptr %1965, 1
  %1968 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1967, i64 0, 2
  %1969 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1968, i64 12, 3, 0
  %1970 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1969, i64 1024, 3, 1
  %1971 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1970, i64 768, 3, 2
  %1972 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1971, i64 786432, 4, 0
  %1973 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1972, i64 768, 4, 1
  %1974 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1973, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %1965, ptr %1122, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9437184), i1 false)
  %1975 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %1976 = ptrtoint ptr %1975 to i64
  %1977 = add i64 %1976, 63
  %1978 = urem i64 %1977, 64
  %1979 = sub i64 %1977, %1978
  %1980 = inttoptr i64 %1979 to ptr
  %1981 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1975, 0
  %1982 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1981, ptr %1980, 1
  %1983 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1982, i64 0, 2
  %1984 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1983, i64 12, 3, 0
  %1985 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1984, i64 1024, 3, 1
  %1986 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1985, i64 768, 3, 2
  %1987 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1986, i64 786432, 4, 0
  %1988 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1987, i64 768, 4, 1
  %1989 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1988, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %1980, ptr %1142, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9437184), i1 false)
  %1990 = add i64 %104, 1
  br label %103

1991:                                             ; preds = %103
  %1992 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %1993 = ptrtoint ptr %1992 to i64
  %1994 = add i64 %1993, 63
  %1995 = urem i64 %1994, 64
  %1996 = sub i64 %1994, %1995
  %1997 = inttoptr i64 %1996 to ptr
  br label %1998

1998:                                             ; preds = %2001, %1991
  %1999 = phi i64 [ %2003, %2001 ], [ 0, %1991 ]
  %2000 = icmp slt i64 %1999, 1
  br i1 %2000, label %2001, label %2004

2001:                                             ; preds = %1998
  %2002 = getelementptr float, ptr %1997, i64 %1999
  store float 0.000000e+00, ptr %2002, align 4
  %2003 = add i64 %1999, 1
  br label %1998

2004:                                             ; preds = %1998
  br label %2005

2005:                                             ; preds = %2030, %2004
  %2006 = phi i64 [ %2031, %2030 ], [ 0, %2004 ]
  %2007 = icmp slt i64 %2006, 768
  br i1 %2007, label %2008, label %2032

2008:                                             ; preds = %2005
  %2009 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %105, 1
  br label %2010

2010:                                             ; preds = %2028, %2008
  %2011 = phi i64 [ %2029, %2028 ], [ 0, %2008 ]
  %2012 = icmp slt i64 %2011, 1
  br i1 %2012, label %2013, label %2030

2013:                                             ; preds = %2010
  br label %2014

2014:                                             ; preds = %2017, %2013
  %2015 = phi i64 [ %2027, %2017 ], [ 0, %2013 ]
  %2016 = icmp slt i64 %2015, 8
  br i1 %2016, label %2017, label %2028

2017:                                             ; preds = %2014
  %2018 = getelementptr float, ptr %2009, i64 %2006
  %2019 = mul i64 %2011, 768
  %2020 = add i64 %2019, %2015
  %2021 = getelementptr float, ptr %2018, i64 %2020
  %2022 = load float, ptr %2021, align 4
  %2023 = getelementptr float, ptr %1997, i64 %2011
  %2024 = load float, ptr %2023, align 4
  %2025 = fmul float %2022, %2022
  %2026 = fadd float %2024, %2025
  store float %2026, ptr %2023, align 4
  %2027 = add i64 %2015, 1
  br label %2014

2028:                                             ; preds = %2014
  %2029 = add i64 %2011, 1
  br label %2010

2030:                                             ; preds = %2010
  %2031 = add i64 %2006, 8
  br label %2005

2032:                                             ; preds = %2005
  %2033 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32000) to i64), i64 64))
  %2034 = ptrtoint ptr %2033 to i64
  %2035 = add i64 %2034, 63
  %2036 = urem i64 %2035, 64
  %2037 = sub i64 %2035, %2036
  %2038 = inttoptr i64 %2037 to ptr
  %2039 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %2033, 0
  %2040 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2039, ptr %2038, 1
  %2041 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2040, i64 0, 2
  %2042 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2041, i64 1, 3, 0
  %2043 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2042, i64 32000, 3, 1
  %2044 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2043, i64 32000, 4, 0
  %2045 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2044, i64 1, 4, 1
  br label %2046

2046:                                             ; preds = %2065, %2032
  %2047 = phi i64 [ %2067, %2065 ], [ 0, %2032 ]
  %2048 = icmp slt i64 %2047, 32000
  br i1 %2048, label %2049, label %2068

2049:                                             ; preds = %2046
  br label %2050

2050:                                             ; preds = %2063, %2049
  %2051 = phi i64 [ %2064, %2063 ], [ 0, %2049 ]
  %2052 = icmp slt i64 %2051, 1
  br i1 %2052, label %2053, label %2065

2053:                                             ; preds = %2050
  br label %2054

2054:                                             ; preds = %2057, %2053
  %2055 = phi i64 [ %2062, %2057 ], [ 0, %2053 ]
  %2056 = icmp slt i64 %2055, 8
  br i1 %2056, label %2057, label %2063

2057:                                             ; preds = %2054
  %2058 = getelementptr float, ptr %2038, i64 %2047
  %2059 = mul i64 %2051, 32000
  %2060 = add i64 %2059, %2055
  %2061 = getelementptr float, ptr %2058, i64 %2060
  store float 0.000000e+00, ptr %2061, align 4
  %2062 = add i64 %2055, 1
  br label %2054

2063:                                             ; preds = %2054
  %2064 = add i64 %2051, 1
  br label %2050

2065:                                             ; preds = %2050
  %2066 = getelementptr float, ptr %2038, i64 %2047
  call void @llvm.memcpy.p0.p0.i64(ptr %2066, ptr %2066, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %2067 = add i64 %2047, 8
  br label %2046

2068:                                             ; preds = %2046
  br label %2069

2069:                                             ; preds = %2118, %2068
  %2070 = phi i64 [ %2119, %2118 ], [ 0, %2068 ]
  %2071 = icmp slt i64 %2070, 32000
  br i1 %2071, label %2072, label %2120

2072:                                             ; preds = %2069
  br label %2073

2073:                                             ; preds = %2115, %2072
  %2074 = phi i64 [ %2117, %2115 ], [ 0, %2072 ]
  %2075 = icmp slt i64 %2074, 768
  br i1 %2075, label %2076, label %2118

2076:                                             ; preds = %2073
  %2077 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %105, 1
  br label %2078

2078:                                             ; preds = %2113, %2076
  %2079 = phi i64 [ %2114, %2113 ], [ 0, %2076 ]
  %2080 = icmp slt i64 %2079, 1
  br i1 %2080, label %2081, label %2115

2081:                                             ; preds = %2078
  br label %2082

2082:                                             ; preds = %2111, %2081
  %2083 = phi i64 [ %2112, %2111 ], [ 0, %2081 ]
  %2084 = icmp slt i64 %2083, 8
  br i1 %2084, label %2085, label %2113

2085:                                             ; preds = %2082
  br label %2086

2086:                                             ; preds = %2089, %2085
  %2087 = phi i64 [ %2110, %2089 ], [ 0, %2085 ]
  %2088 = icmp slt i64 %2087, 8
  br i1 %2088, label %2089, label %2111

2089:                                             ; preds = %2086
  %2090 = getelementptr float, ptr %2077, i64 %2074
  %2091 = mul i64 %2079, 768
  %2092 = add i64 %2091, %2087
  %2093 = getelementptr float, ptr %2090, i64 %2092
  %2094 = load float, ptr %2093, align 4
  %2095 = getelementptr float, ptr %1997, i64 %2079
  %2096 = load float, ptr %2095, align 4
  %2097 = getelementptr float, ptr %2038, i64 %2070
  %2098 = mul i64 %2079, 32000
  %2099 = add i64 %2098, %2083
  %2100 = getelementptr float, ptr %2097, i64 %2099
  %2101 = load float, ptr %2100, align 4
  %2102 = fdiv float %2096, 7.680000e+02
  %2103 = fadd float %2102, 0x3EE4F8B580000000
  %2104 = call float @llvm.sqrt.f32(float %2103)
  %2105 = fdiv float 1.000000e+00, %2104
  %2106 = fmul float %2094, %2105
  %2107 = fmul float %2106, 1.200000e+01
  %2108 = fmul float %2107, 1.300000e+01
  %2109 = fadd float %2101, %2108
  store float %2109, ptr %2100, align 4
  %2110 = add i64 %2087, 1
  br label %2086

2111:                                             ; preds = %2086
  %2112 = add i64 %2083, 1
  br label %2082

2113:                                             ; preds = %2082
  %2114 = add i64 %2079, 1
  br label %2078

2115:                                             ; preds = %2078
  %2116 = getelementptr float, ptr %2038, i64 %2070
  call void @llvm.memcpy.p0.p0.i64(ptr %2116, ptr %2116, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %2117 = add i64 %2074, 8
  br label %2073

2118:                                             ; preds = %2073
  %2119 = add i64 %2070, 8
  br label %2069

2120:                                             ; preds = %2069
  %2121 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %2122 = ptrtoint ptr %2121 to i64
  %2123 = add i64 %2122, 63
  %2124 = urem i64 %2123, 64
  %2125 = sub i64 %2123, %2124
  %2126 = inttoptr i64 %2125 to ptr
  br label %2127

2127:                                             ; preds = %2130, %2120
  %2128 = phi i64 [ %2132, %2130 ], [ 0, %2120 ]
  %2129 = icmp slt i64 %2128, 1
  br i1 %2129, label %2130, label %2133

2130:                                             ; preds = %2127
  %2131 = getelementptr float, ptr %2126, i64 %2128
  store float 0xFFF0000000000000, ptr %2131, align 4
  %2132 = add i64 %2128, 1
  br label %2127

2133:                                             ; preds = %2127
  %2134 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), i64 64))
  %2135 = ptrtoint ptr %2134 to i64
  %2136 = add i64 %2135, 63
  %2137 = urem i64 %2136, 64
  %2138 = sub i64 %2136, %2137
  %2139 = inttoptr i64 %2138 to ptr
  br label %2140

2140:                                             ; preds = %2143, %2133
  %2141 = phi i64 [ %2145, %2143 ], [ 0, %2133 ]
  %2142 = icmp slt i64 %2141, 1
  br i1 %2142, label %2143, label %2146

2143:                                             ; preds = %2140
  %2144 = getelementptr i64, ptr %2139, i64 %2141
  store i64 0, ptr %2144, align 4
  %2145 = add i64 %2141, 1
  br label %2140

2146:                                             ; preds = %2140
  br label %2147

2147:                                             ; preds = %2175, %2146
  %2148 = phi i64 [ %2176, %2175 ], [ 0, %2146 ]
  %2149 = icmp slt i64 %2148, 32000
  br i1 %2149, label %2150, label %2177

2150:                                             ; preds = %2147
  br label %2151

2151:                                             ; preds = %2173, %2150
  %2152 = phi i64 [ %2174, %2173 ], [ 0, %2150 ]
  %2153 = icmp slt i64 %2152, 1
  br i1 %2153, label %2154, label %2175

2154:                                             ; preds = %2151
  br label %2155

2155:                                             ; preds = %2158, %2154
  %2156 = phi i64 [ %2172, %2158 ], [ 0, %2154 ]
  %2157 = icmp slt i64 %2156, 8
  br i1 %2157, label %2158, label %2173

2158:                                             ; preds = %2155
  %2159 = getelementptr float, ptr %2038, i64 %2148
  %2160 = mul i64 %2152, 32000
  %2161 = add i64 %2160, %2156
  %2162 = getelementptr float, ptr %2159, i64 %2161
  %2163 = load float, ptr %2162, align 4
  %2164 = getelementptr float, ptr %2126, i64 %2152
  %2165 = load float, ptr %2164, align 4
  %2166 = getelementptr i64, ptr %2139, i64 %2152
  %2167 = load i64, ptr %2166, align 4
  %2168 = add i64 %2156, %2148
  %2169 = fcmp ogt float %2163, %2165
  %2170 = select i1 %2169, float %2163, float %2165
  %2171 = select i1 %2169, i64 %2168, i64 %2167
  store float %2170, ptr %2164, align 4
  store i64 %2171, ptr %2166, align 4
  %2172 = add i64 %2156, 1
  br label %2155

2173:                                             ; preds = %2155
  %2174 = add i64 %2152, 1
  br label %2151

2175:                                             ; preds = %2151
  %2176 = add i64 %2148, 8
  br label %2147

2177:                                             ; preds = %2147
  %2178 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %2045, ptr %2178, align 8
  call void @printMemrefF32(i64 2, ptr %2178)
  %2179 = add i64 %87, 1
  br label %31

2180:                                             ; preds = %31
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
declare i64 @llvm.smin.i64(i64, i64) #1

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
