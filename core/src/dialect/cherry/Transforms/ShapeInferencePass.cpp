#include "dialect/cherry/IR/CherryDialect.h"
#include "dialect/cherry/IR/CherryTypes.h"
#include "dialect/cherry/Transforms/Passes.h"
#include "interfaces/CherryInterface.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
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
    auto                                    funcOp = getOperation();
    llvm::SmallPtrSet<mlir::Operation*, 16> opWorklist;
    funcOp->walk([&](mlir::Operation* op) -> void {
        bool isDynamic = llvm::any_of(op->getResultTypes(), [](Type resultType) {
            if (auto cherryType = llvm::dyn_cast<mlir::cherry::CherryTensorType>(resultType)) {
                return cherryType.isDynamic();
            }
            return false;
        });
        if (isDynamic) {
            opWorklist.insert(op);
        }
    });

    while (!opWorklist.empty()) {
        auto nextop = llvm::find_if(opWorklist, [](Operation* op) {
            bool inputsReady = llvm::all_of(op->getOperandTypes(), [](Type operandType) {
                if (auto cherryType = llvm::dyn_cast<mlir::cherry::CherryTensorType>(operandType)) {
                    return !cherryType.isDynamic();
                }
                return true;
            });
            return inputsReady;
        });
        if (nextop == opWorklist.end()) {
            break;
        }
        Operation* op = *nextop;
        opWorklist.erase(op);

        if (auto shapeInferenceOp = llvm::dyn_cast<ShapeInference>(op)) {
            shapeInferenceOp.inferShapes();
        }
        else {
            op->emitError("unable to infer shape of operation without shape "
                          "inference interface");
            return signalPassFailure();
        }
    }
    if (!opWorklist.empty()) {
        funcOp.emitError("Shape inference failed, ")
            << opWorklist.size() << " operations couldn't be inferred\n";
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::cherry::createCherryShapeInferencePass()
{
    return std::make_unique<ShapeInferencePass>();
}
