#ifndef DIALECT_CHERRY_CONVERSION_PASS_TD
#define DIALECT_CHERRY_CONVERSION_PASS_TD

include "mlir/Pass/PassBase.td"
def ConvertCherryToLinalgPass : Pass<"convert-cherry-to-linalg", "mlir::ModuleOp">{
    let summary = "Convert Cherry To Linalg Pass";
    let dependentDialects = [
        "::mlir::cherry::CherryDialect",
        "::mlir::linalg::LinalgDialect",
        "::mlir::tensor::TensorDialect",
    ];
   // let constructor = "mlir::cherry::createConvertCherryToLinalgPass()";

}


#endif
