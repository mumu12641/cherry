; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__constant_1x2x2xf32 = private constant [1 x [2 x [2 x float]]] [[2 x [2 x float]] [[2 x float] [float 1.000000e+00, float 2.000000e+00], [2 x float] [float 3.000000e+00, float 4.000000e+00]]], align 64

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

declare void @printMemrefF32(i64, ptr)

define void @host() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2) to i64), i64 64))
  %8 = ptrtoint ptr %7 to i64
  %9 = add i64 %8, 63
  %10 = urem i64 %9, 64
  %11 = sub i64 %9, %10
  %12 = inttoptr i64 %11 to ptr
  br label %13

13:                                               ; preds = %16, %0
  %14 = phi i64 [ %25, %16 ], [ 0, %0 ]
  %15 = icmp slt i64 %14, 2
  br i1 %15, label %16, label %26

16:                                               ; preds = %13
  %17 = uitofp i64 %14 to float
  %18 = fmul float %17, -2.000000e+00
  %19 = fdiv float %18, 4.000000e+00
  %20 = call float @llvm.pow.f32(float 1.000000e+04, float %19)
  %21 = call float @llvm.cos.f32(float %20)
  %22 = call float @llvm.sin.f32(float %20)
  %23 = getelementptr float, ptr %6, i64 %14
  store float %21, ptr %23, align 4
  %24 = getelementptr float, ptr %12, i64 %14
  store float %22, ptr %24, align 4
  %25 = add i64 %14, 1
  br label %13

26:                                               ; preds = %13
  %27 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2) to i64), i64 64))
  %28 = ptrtoint ptr %27 to i64
  %29 = add i64 %28, 63
  %30 = urem i64 %29, 64
  %31 = sub i64 %29, %30
  %32 = inttoptr i64 %31 to ptr
  %33 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %27, 0
  %34 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, ptr %32, 1
  %35 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, i64 0, 2
  %36 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %35, i64 1, 3, 0
  %37 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, i64 2, 3, 1
  %38 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %37, i64 1, 3, 2
  %39 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, i64 2, 4, 0
  %40 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, i64 1, 4, 1
  %41 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %40, i64 1, 4, 2
  %42 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2) to i64), i64 64))
  %43 = ptrtoint ptr %42 to i64
  %44 = add i64 %43, 63
  %45 = urem i64 %44, 64
  %46 = sub i64 %44, %45
  %47 = inttoptr i64 %46 to ptr
  %48 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %42, 0
  %49 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %48, ptr %47, 1
  %50 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, i64 0, 2
  %51 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %50, i64 1, 3, 0
  %52 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %51, i64 2, 3, 1
  %53 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %52, i64 1, 3, 2
  %54 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %53, i64 2, 4, 0
  %55 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %54, i64 1, 4, 1
  %56 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, i64 1, 4, 2
  br label %57

57:                                               ; preds = %96, %26
  %58 = phi i64 [ %97, %96 ], [ 0, %26 ]
  %59 = icmp slt i64 %58, 1
  br i1 %59, label %60, label %98

60:                                               ; preds = %57
  br label %61

61:                                               ; preds = %94, %60
  %62 = phi i64 [ %95, %94 ], [ 0, %60 ]
  %63 = icmp slt i64 %62, 2
  br i1 %63, label %64, label %96

64:                                               ; preds = %61
  br label %65

65:                                               ; preds = %68, %64
  %66 = phi i64 [ %93, %68 ], [ 0, %64 ]
  %67 = icmp slt i64 %66, 1
  br i1 %67, label %68, label %94

68:                                               ; preds = %65
  %69 = mul i64 %58, 4
  %70 = mul i64 %62, 2
  %71 = add i64 %69, %70
  %72 = add i64 %71, %66
  %73 = getelementptr float, ptr @__constant_1x2x2xf32, i64 %72
  %74 = load float, ptr %73, align 4
  %75 = getelementptr float, ptr getelementptr (float, ptr @__constant_1x2x2xf32, i32 1), i64 %72
  %76 = load float, ptr %75, align 4
  %77 = add i64 %62, %66
  %78 = getelementptr float, ptr %6, i64 %77
  %79 = load float, ptr %78, align 4
  %80 = getelementptr float, ptr %12, i64 %77
  %81 = load float, ptr %80, align 4
  %82 = fmul float %74, %79
  %83 = fmul float %76, %81
  %84 = fsub float %82, %83
  %85 = fmul float %76, %79
  %86 = fmul float %74, %81
  %87 = fadd float %85, %86
  %88 = mul i64 %58, 2
  %89 = add i64 %88, %62
  %90 = add i64 %89, %66
  %91 = getelementptr float, ptr %32, i64 %90
  store float %84, ptr %91, align 4
  %92 = getelementptr float, ptr %47, i64 %90
  store float %87, ptr %92, align 4
  %93 = add i64 %66, 1
  br label %65

94:                                               ; preds = %65
  %95 = add i64 %62, 1
  br label %61

96:                                               ; preds = %61
  %97 = add i64 %58, 1
  br label %57

98:                                               ; preds = %57
  %99 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4) to i64), i64 64))
  %100 = ptrtoint ptr %99 to i64
  %101 = add i64 %100, 63
  %102 = urem i64 %101, 64
  %103 = sub i64 %101, %102
  %104 = inttoptr i64 %103 to ptr
  %105 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %99, 0
  %106 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %105, ptr %104, 1
  %107 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, i64 0, 2
  %108 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %107, i64 1, 3, 0
  %109 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %108, i64 4, 4, 0
  %110 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %109, i64 2, 3, 1
  %111 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %110, i64 2, 4, 1
  %112 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %111, i64 1, 3, 2
  %113 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %112, i64 1, 4, 2
  %114 = call ptr @llvm.stacksave.p0()
  %115 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %41, ptr %115, align 8
  %116 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %115, 1
  %117 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %113, ptr %117, align 8
  %118 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %117, 1
  %119 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %116, ptr %119, align 8
  %120 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %118, ptr %120, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %119, ptr %120)
  call void @llvm.stackrestore.p0(ptr %114)
  %121 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, i64 1, 2
  %122 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %121, i64 1, 3, 0
  %123 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %122, i64 4, 4, 0
  %124 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %123, i64 2, 3, 1
  %125 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %124, i64 2, 4, 1
  %126 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %125, i64 1, 3, 2
  %127 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, i64 1, 4, 2
  %128 = call ptr @llvm.stacksave.p0()
  %129 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %56, ptr %129, align 8
  %130 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %129, 1
  %131 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %127, ptr %131, align 8
  %132 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %131, 1
  %133 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %130, ptr %133, align 8
  %134 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %132, ptr %134, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %133, ptr %134)
  call void @llvm.stackrestore.p0(ptr %128)
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %99, 0
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, ptr %104, 1
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 0, 2
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 1, 3, 0
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, i64 4, 4, 0
  %140 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %139, i64 4, 3, 1
  %141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %140, i64 1, 4, 1
  %142 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, ptr %142, align 8
  call void @printMemrefF32(i64 2, ptr %142)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.pow.f32(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.cos.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sin.f32(float) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
