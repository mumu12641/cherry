#include "Conversion/CherryToLinalg/LinalgVectorization.h"

#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::cherry {
#define GEN_PASS_DEF_CHERRYLINALGVECTORIZATIONPASS
#include "Conversion/Passes.h.inc"
}   // namespace mlir::cherry

using namespace mlir;
using namespace mlir::cherry;
using namespace mlir::linalg;
struct CherryLinalgVectorizationPass
    : mlir::cherry::impl::CherryLinalgVectorizationPassBase<CherryLinalgVectorizationPass>
{
    using mlir::cherry::impl::CherryLinalgVectorizationPassBase<
        CherryLinalgVectorizationPass>::CherryLinalgVectorizationPassBase;
    void runOnOperation() override;
};

void CherryLinalgVectorizationPass::runOnOperation()
{
    func::FuncOp      funcOp = getOperation();
    MLIRContext*      ctx    = &getContext();
    RewritePatternSet patterns(ctx);

    // A. 添加在这个文件里没有的、我们自定义的通用向量化 Pattern
    // 这负责处理 matmul, generic, pooling 等
    patterns.add<GenericVectorizationPattern>(ctx);

    // B. 添加 LLVM 原生提供的辅助 Pattern (可选，但推荐加上)
    // 处理 tensor.pad 的向量化
    mlir::linalg::populatePadOpVectorizationPatterns(patterns);
    // 处理卷积的特殊优化 (虽然通用 Pattern 也能处理，但专用 Pattern 往往生成的代码更好)
    mlir::linalg::populateConvolutionVectorizationPatterns(patterns);

    // 使用贪婪重写驱动应用 Pattern
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
    }
    // auto func = getOperation();
    // auto context = func.getContext();
    // auto builder = OpBuilder::atBlockBegin(func.getBody());

    // // Vectorize Cherry operations to Linalg operations.
    // auto vectorizationPatterns = CherryLinalgVectorizationPatterns(context);
    // if (failed(applyPatternsAndFoldGreedily(func, vectorizationPatterns))) {
    //     signalPassFailure();
    // }
}
