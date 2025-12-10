; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__constant_2x2xf32 = private constant [2 x [2 x float]] [[2 x float] [float 1.000000e+00, float 1.000000e+01], [2 x float] [float 1.000000e+00, float 1.000000e+01]], align 64
@__constant_1x2xf32 = private constant [1 x [2 x float]] [[2 x float] [float 1.000000e+00, float 2.000000e+00]], align 64

declare ptr @malloc(i64)

declare void @printMemrefF32(i64, ptr)

define void @host() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, i64 0, 2
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 1, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 2, 3, 1
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 2, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 1, 4, 1
  br label %14

14:                                               ; preds = %26, %0
  %15 = phi i64 [ %27, %26 ], [ 0, %0 ]
  %16 = icmp slt i64 %15, 1
  br i1 %16, label %17, label %28

17:                                               ; preds = %14
  br label %18

18:                                               ; preds = %21, %17
  %19 = phi i64 [ %25, %21 ], [ 0, %17 ]
  %20 = icmp slt i64 %19, 2
  br i1 %20, label %21, label %26

21:                                               ; preds = %18
  %22 = mul i64 %15, 2
  %23 = add i64 %22, %19
  %24 = getelementptr float, ptr %6, i64 %23
  store float 0.000000e+00, ptr %24, align 4
  %25 = add i64 %19, 1
  br label %18

26:                                               ; preds = %18
  %27 = add i64 %15, 1
  br label %14

28:                                               ; preds = %14
  br label %29

29:                                               ; preds = %61, %28
  %30 = phi i64 [ %62, %61 ], [ 0, %28 ]
  %31 = icmp slt i64 %30, 1
  br i1 %31, label %32, label %63

32:                                               ; preds = %29
  br label %33

33:                                               ; preds = %59, %32
  %34 = phi i64 [ %60, %59 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 2
  br i1 %35, label %36, label %61

36:                                               ; preds = %33
  br label %37

37:                                               ; preds = %40, %36
  %38 = phi i64 [ %58, %40 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 2
  br i1 %39, label %40, label %59

40:                                               ; preds = %37
  %41 = mul i64 %30, 2
  %42 = add i64 %41, %38
  %43 = getelementptr float, ptr @__constant_1x2xf32, i64 %42
  %44 = load float, ptr %43, align 4
  %45 = mul i64 %38, 2
  %46 = add i64 %45, %34
  %47 = getelementptr float, ptr @__constant_2x2xf32, i64 %46
  %48 = load float, ptr %47, align 4
  %49 = add i64 %41, %34
  %50 = getelementptr float, ptr %6, i64 %49
  %51 = load float, ptr %50, align 4
  %52 = icmp sle i64 %34, 0
  %53 = select i1 %52, float 1.000000e+00, float 0.000000e+00
  %54 = fmul float %44, %48
  %55 = fadd float %51, %54
  %56 = fcmp ugt float %53, 5.000000e-01
  %57 = select i1 %56, float %55, float -1.000000e+09
  store float %57, ptr %50, align 4
  %58 = add i64 %38, 1
  br label %37

59:                                               ; preds = %37
  %60 = add i64 %34, 1
  br label %33

61:                                               ; preds = %33
  %62 = add i64 %30, 1
  br label %29

63:                                               ; preds = %29
  %64 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, ptr %64, align 8
  call void @printMemrefF32(i64 2, ptr %64)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
