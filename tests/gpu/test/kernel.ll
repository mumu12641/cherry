; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @square_kernel(i64 %0, i64 %1, ptr %2, ptr %3) {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = sext i32 %5 to i64
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = sext i32 %7 to i64
  %9 = add i64 %0, %6
  %10 = add i64 %1, %8
  %11 = mul i64 %9, 10
  %12 = add i64 %11, %10
  %13 = getelementptr float, ptr %2, i64 %12
  %14 = load float, ptr %13, align 4
  %15 = fmul float %14, %14
  %16 = mul i64 %9, 10
  %17 = add i64 %16, %10
  %18 = getelementptr float, ptr %3, i64 %17
  store float %15, ptr %18, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @square_kernel, !"kernel", i32 1}
