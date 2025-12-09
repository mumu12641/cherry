; ModuleID = 'Driver'
source_filename = "driver.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @printf(ptr, ...)

; ==========================================================
; 1. 修正声明：必须完全匹配 host 的返回值类型
; ==========================================================
declare { ptr, ptr, i64, [3 x i64], [3 x i64] } @host()

; ==========================================================
; print_memref_f32 (Rank 2)
; 注意：你的 host 返回的是 Rank 3，但内部打印调用的是 Rank 2
; 这里只负责实现 host 内部需要的那个打印函数
; ==========================================================
@fmt_val = private constant [4 x i8] c"%f \00"
@fmt_nl  = private constant [2 x i8] c"\0A\00"
@msg_start = private constant [22 x i8] c"Printing MemRef(2d):\0A\00"

define void @print_memref_f32(i64 %rank, ptr %desc_ptr) {
entry:
  call i32 (ptr, ...) @printf(ptr @msg_start)
  %ptr_addr = getelementptr { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %desc_ptr, i32 0, i32 1
  %aligned_ptr = load ptr, ptr %ptr_addr
  %offset_addr = getelementptr { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %desc_ptr, i32 0, i32 2
  %offset = load i64, ptr %offset_addr
  %size0_addr = getelementptr { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %desc_ptr, i32 0, i32 3, i32 0
  %size0 = load i64, ptr %size0_addr
  %size1_addr = getelementptr { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %desc_ptr, i32 0, i32 3, i32 1
  %size1 = load i64, ptr %size1_addr
  %stride0_addr = getelementptr { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %desc_ptr, i32 0, i32 4, i32 0
  %stride0 = load i64, ptr %stride0_addr
  %stride1_addr = getelementptr { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %desc_ptr, i32 0, i32 4, i32 1
  %stride1 = load i64, ptr %stride1_addr

  br label %loop_row_check

loop_row_check:
  %i = phi i64 [ 0, %entry ], [ %i_next, %loop_row_end ]
  %row_cond = icmp slt i64 %i, %size0
  br i1 %row_cond, label %loop_col_check, label %func_end

loop_col_check:
  %j = phi i64 [ 0, %loop_row_check ], [ %j_next, %loop_col_body ]
  %col_cond = icmp slt i64 %j, %size1
  br i1 %col_cond, label %loop_col_body, label %loop_row_end

loop_col_body:
  %term1 = mul i64 %i, %stride0
  %term2 = mul i64 %j, %stride1
  %idx_tmp = add i64 %term1, %term2
  %final_idx = add i64 %idx_tmp, %offset
  %val_ptr = getelementptr float, ptr %aligned_ptr, i64 %final_idx
  %val = load float, ptr %val_ptr
  %val_d = fpext float %val to double
  call i32 (ptr, ...) @printf(ptr @fmt_val, double %val_d)
  %j_next = add i64 %j, 1
  br label %loop_col_check

loop_row_end:
  call i32 (ptr, ...) @printf(ptr @fmt_nl)
  %i_next = add i64 %i, 1
  br label %loop_row_check

func_end:
  ret void
}

; ==========================================================
; Main 函数
; ==========================================================
define i32 @main() {
  ; 2. 修正调用：接收返回值（即使我们不使用它）
  ; 这样编译器才会正确分配栈空间来接收那个巨大的结构体
  %unused_result = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @host()
  
  ret i32 0
}
