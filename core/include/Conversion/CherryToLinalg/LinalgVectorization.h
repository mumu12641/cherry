#ifndef DIALECT_LINALG_VECTORIZATION_H
#define DIALECT_LINALG_VECTORIZATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include <memory>

namespace mlir::cherry {
#define GEN_PASS_DECL_CHERRYLINALGVECTORIZATIONPASS
#include "Conversion/Passes.h.inc"
struct GenericVectorizationPattern : public OpInterfaceRewritePattern<mlir::linalg::LinalgOp>
{
    using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

    LogicalResult matchAndRewrite(mlir::linalg::LinalgOp op,
                                  PatternRewriter&       rewriter) const override
    {
        return mlir::linalg::vectorize(rewriter,
                                       op,
                                       /*inputVectorSizes=*/{},
                                       /*inputScalableVecDims=*/{},
                                       /*vectorizeNDExtract=*/false,
                                       /*flatten1DDepthwiseConv=*/false);
    }
};
}   // namespace mlir::cherry
#endif
