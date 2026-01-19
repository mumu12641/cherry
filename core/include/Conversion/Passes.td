#ifndef DIALECT_CHERRY_CONVERSION_PASS_TD
#define DIALECT_CHERRY_CONVERSION_PASS_TD

include "mlir/Pass/PassBase.td"
def ConvertCherryToLinalgPass : Pass<"convert-cherry-to-linalg", "mlir::ModuleOp">{
    let summary = "Convert Cherry To Linalg Pass";
    let dependentDialects = [
        "::mlir::cherry::CherryDialect",
        "::mlir::linalg::LinalgDialect",
        "::mlir::tensor::TensorDialect",
        "::mlir::func::FuncDialect",
    ];
}

def CherryLinalgTilingPass : Pass<"cherry-linalg-tiling", "mlir::func::FuncOp">{
    let summary = "Linalg Tiling Pass";
    let dependentDialects = [
        "::mlir::cherry::CherryDialect",
        "::mlir::linalg::LinalgDialect",
        "::mlir::func::FuncDialect",
    ];
    
    let options = [
        Option<"tileSize", "tile-size", "int64_t", /*default=*/"32", 
               "Tile size for linalg operations">,
        Option<"outerTileSize", "outer-tile-size", "int64_t", /*default=*/"128", 
               "Outer tile size for linalg operations">,
        Option<"hierarchical", "hierarchical", "bool", /*default=*/"true", 
               "Enable hierarchical tiling">,
        Option<"parallelLoops", "parallel-loops", "bool", /*default=*/"false", 
               "Enable parallel loops">
    ];
}

def CherryLinalgVectorizationPass : Pass<"cherry-linalg-vectorization", "mlir::func::FuncOp">{
    let summary = "Linalg Vectorization Pass";
    let dependentDialects = [
        "::mlir::cherry::CherryDialect",
        "::mlir::linalg::LinalgDialect",
        "::mlir::func::FuncDialect",
    ];
}

def CherryInsertHostRegisterPass : Pass<"cherry-insert-host-register", "mlir::func::FuncOp">{
    let summary = "Insert Host Register Pass";
    let dependentDialects = [
        "::mlir::gpu::GPUDialect",
        "::mlir::func::FuncDialect",
    ];
}

#endif
