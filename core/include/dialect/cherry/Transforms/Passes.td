#ifndef DIALECT_CHERRY_PASSES_TD
#define DIALECT_CHERRY_PASSES_TD

include "mlir/Pass/PassBase.td"

def ShapeInferencePass : Pass<"ShapeInferencePass", "mlir::cherry::FuncOp">{
    let summary = "Tensor shape infer pass";
    let dependentDialects = [
        "::mlir::cherry::CherryDialect",
    ];
    let constructor = "mlir::cherry::createCherryShapeInferencePass()";
}

#endif

