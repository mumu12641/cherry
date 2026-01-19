module {
      llvm.func @square_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, nvvm.kernel} {
        %0 = llvm.mlir.constant(10 : index) : i64
        %1 = nvvm.read.ptx.sreg.ctaid.x : i32
        %2 = llvm.sext %1 : i32 to i64
        %3 = nvvm.read.ptx.sreg.tid.x : i32
        %4 = llvm.sext %3 : i32 to i64
        %5 = llvm.add %arg0, %2 : i64
        %6 = llvm.add %arg1, %4 : i64
        %7 = llvm.mul %5, %0 : i64
        %8 = llvm.add %7, %6 : i64
        %9 = llvm.getelementptr %arg2[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %10 = llvm.load %9 : !llvm.ptr -> f32
        %11 = llvm.fmul %10, %10 : f32
        %12 = llvm.mul %5, %0 : i64
        %13 = llvm.add %12, %6 : i64
        %14 = llvm.getelementptr %arg3[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %11, %14 : f32, !llvm.ptr
        llvm.return
      }
}
