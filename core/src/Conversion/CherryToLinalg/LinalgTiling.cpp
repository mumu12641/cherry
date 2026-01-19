
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
    int64_t            l1Size = tileSize.getValue();
    int64_t            l2Size = outerTileSize.getValue();

    LinalgTilingLoopType loopType = parallelLoops.getValue() ? LinalgTilingLoopType::ParallelLoops
                                                             : LinalgTilingLoopType::Loops;
    llvm::SmallVector<mlir::linalg::LinalgOp> worklist;
    funcOp.walk([&](mlir::linalg::LinalgOp op) { worklist.push_back(op); });

    auto applyTiling =
        [&](mlir::linalg::LinalgOp op, int64_t size, LinalgTilingLoopType lt) -> LogicalResult {
        unsigned numLoops = op.getNumLoops();
        if (numLoops == 0) return success();
        SmallVector<int64_t> tileSizes(numLoops, size);
        LinalgTilingOptions  tilingOptions;
        tilingOptions.setTileSizes(tileSizes).setLoopType(lt);

        rewriter.setInsertionPoint(op);
        FailureOr<mlir::linalg::TiledLinalgOp> tiledOp =
            mlir::linalg::tileLinalgOp(rewriter, op, tilingOptions);

        if (succeeded(tiledOp)) {
            rewriter.replaceOp(op, tiledOp->tensorResults);
            return success();
        }
        return failure();
    };

    if (hierarchical.getValue()) {
        for (LinalgOp op : worklist) {
            if (op.getNumReductionLoops() > 0) {
                if (failed(applyTiling(op, l2Size, loopType))) {
                    continue;
                }
            }
        }
        worklist.clear();
        funcOp.walk([&](LinalgOp op) { worklist.push_back(op); });

        for (LinalgOp op : worklist) {
            if (failed(applyTiling(op, l1Size, LinalgTilingLoopType::Loops))) {
                signalPassFailure();
                return;
            }
        }
    }
    else {
        for (LinalgOp op : worklist) {
            if (failed(applyTiling(op, l1Size, LinalgTilingLoopType::Loops))) {
                signalPassFailure();
                return;
            }
        }
    }
}
