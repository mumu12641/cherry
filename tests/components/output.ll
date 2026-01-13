; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @printMemrefF32(i64, ptr)

define void @host() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %8 = ptrtoint ptr %7 to i64
  %9 = add i64 %8, 63
  %10 = urem i64 %9, 64
  %11 = sub i64 %9, %10
  %12 = inttoptr i64 %11 to ptr
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %7, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, ptr %12, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 0, 2
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 1, 3, 0
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 768, 3, 1
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 768, 4, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 1, 4, 1
  br label %20

20:                                               ; preds = %39, %0
  %21 = phi i64 [ %41, %39 ], [ 0, %0 ]
  %22 = icmp slt i64 %21, 768
  br i1 %22, label %23, label %42

23:                                               ; preds = %20
  br label %24

24:                                               ; preds = %37, %23
  %25 = phi i64 [ %38, %37 ], [ 0, %23 ]
  %26 = icmp slt i64 %25, 1
  br i1 %26, label %27, label %39

27:                                               ; preds = %24
  br label %28

28:                                               ; preds = %31, %27
  %29 = phi i64 [ %36, %31 ], [ 0, %27 ]
  %30 = icmp slt i64 %29, 8
  br i1 %30, label %31, label %37

31:                                               ; preds = %28
  %32 = getelementptr float, ptr %12, i64 %21
  %33 = mul i64 %25, 768
  %34 = add i64 %33, %29
  %35 = getelementptr float, ptr %32, i64 %34
  store float 0.000000e+00, ptr %35, align 4
  %36 = add i64 %29, 1
  br label %28

37:                                               ; preds = %28
  %38 = add i64 %25, 1
  br label %24

39:                                               ; preds = %24
  %40 = getelementptr float, ptr %12, i64 %21
  call void @llvm.memcpy.p0.p0.i64(ptr %40, ptr %40, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %41 = add i64 %21, 8
  br label %20

42:                                               ; preds = %20
  br label %43

43:                                               ; preds = %81, %42
  %44 = phi i64 [ %82, %81 ], [ 0, %42 ]
  %45 = icmp slt i64 %44, 768
  br i1 %45, label %46, label %83

46:                                               ; preds = %43
  br label %47

47:                                               ; preds = %77, %46
  %48 = phi i64 [ %80, %77 ], [ 0, %46 ]
  %49 = icmp slt i64 %48, 768
  br i1 %49, label %50, label %81

50:                                               ; preds = %47
  br label %51

51:                                               ; preds = %75, %50
  %52 = phi i64 [ %76, %75 ], [ 0, %50 ]
  %53 = icmp slt i64 %52, 1
  br i1 %53, label %54, label %77

54:                                               ; preds = %51
  br label %55

55:                                               ; preds = %73, %54
  %56 = phi i64 [ %74, %73 ], [ 0, %54 ]
  %57 = icmp slt i64 %56, 8
  br i1 %57, label %58, label %75

58:                                               ; preds = %55
  br label %59

59:                                               ; preds = %62, %58
  %60 = phi i64 [ %72, %62 ], [ 0, %58 ]
  %61 = icmp slt i64 %60, 8
  br i1 %61, label %62, label %73

62:                                               ; preds = %59
  %63 = getelementptr float, ptr %12, i64 %44
  %64 = mul i64 %52, 768
  %65 = add i64 %64, %56
  %66 = getelementptr float, ptr %63, i64 %65
  %67 = load float, ptr %66, align 4
  %68 = fadd float %67, 0x3FBC28F5E0000000
  %69 = getelementptr float, ptr %6, i64 %48
  %70 = add i64 %64, %60
  %71 = getelementptr float, ptr %69, i64 %70
  store float 0x3FF19999A0000000, ptr %71, align 4
  store float %68, ptr %66, align 4
  %72 = add i64 %60, 1
  br label %59

73:                                               ; preds = %59
  %74 = add i64 %56, 1
  br label %55

75:                                               ; preds = %55
  %76 = add i64 %52, 1
  br label %51

77:                                               ; preds = %51
  %78 = getelementptr float, ptr %6, i64 %48
  call void @llvm.memcpy.p0.p0.i64(ptr %78, ptr %78, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %79 = getelementptr float, ptr %12, i64 %44
  call void @llvm.memcpy.p0.p0.i64(ptr %79, ptr %79, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 8), i1 false)
  %80 = add i64 %48, 8
  br label %47

81:                                               ; preds = %47
  %82 = add i64 %44, 8
  br label %43

83:                                               ; preds = %43
  %84 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, ptr %84, align 8
  call void @printMemrefF32(i64 2, ptr %84)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
