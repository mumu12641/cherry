#include "dialect/cherry/IR/CherryDialect.h"
#include "dialect/cherry/Transforms/Passes.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::cherry {
#define GEN_PASS_DEF_SHAPEINFERENCEPASS
#include "dialect/cherry/Transforms/Passes.h.inc"
}   // namespace mlir::cherry

using namespace ::mlir;
using namespace ::mlir::cherry;

struct ShapeInferencePass : mlir::cherry::impl::ShapeInferencePassBase<ShapeInferencePass>
{
    using mlir::cherry::impl::ShapeInferencePassBase<ShapeInferencePass>::ShapeInferencePassBase;
    void runOnOperation() override;
};

void ShapeInferencePass::runOnOperation()
{
    llvm::outs() << "ShapeInferencePass::runOnOperation()" << "\n";
}

std::unique_ptr<mlir::Pass> mlir::cherry::createCherryShapeInferencePass()
{
    return std::make_unique<ShapeInferencePass>();
}