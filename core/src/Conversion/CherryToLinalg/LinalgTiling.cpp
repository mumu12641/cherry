
#include "Conversion/CherryToLinalg/LinalgTiling.h"

#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::cherry {
#define GEN_PASS_DEF_CHERRYLINALGTILINGPASS
#include "Conversion/Passes.h.inc"
}   // namespace mlir::cherry

using namespace mlir;
using namespace mlir::cherry;
using namespace mlir::linalg;

struct CherryLinalgTilingPass
    : mlir::cherry::impl::CherryLinalgTilingPassBase<CherryLinalgTilingPass>
{
    using mlir::cherry::impl::CherryLinalgTilingPassBase<
        CherryLinalgTilingPass>::CherryLinalgTilingPassBase;
    void runOnOperation() override;
};

void CherryLinalgTilingPass::runOnOperation()
{

    mlir::func::FuncOp funcOp = getOperation();
    mlir::IRRewriter   rewriter(&getContext());

    LinalgTilingLoopType loopType = parallelLoops.getValue() ? LinalgTilingLoopType::ParallelLoops
                                                             : LinalgTilingLoopType::Loops;
    llvm::SmallVector<mlir::linalg::LinalgOp> worklist;
    funcOp.walk([&](mlir::linalg::LinalgOp op) { worklist.push_back(op); });

    llvm::SmallVector<int64_t, 3> innerTileSizes(3, tileSize.getValue());
    llvm::SmallVector<int64_t, 3> outerTileSizes(3, outerTileSize.getValue());

    mlir::linalg::LinalgTilingOptions options;
    options.setTileSizes(innerTileSizes);
    options.setLoopType(loopType);
    for (mlir::linalg::LinalgOp op : worklist) {
        rewriter.setInsertionPoint(op);
        mlir::FailureOr<mlir::linalg::TiledLinalgOp> tiledResult =
            mlir::linalg::tileLinalgOp(rewriter, op, options);
        rewriter.replaceOp(op, tiledResult->tensorResults);
    }
}
