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
    llvm::outs() << "ShapeInferencePass::runOnOperation()" << "\n";
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
            llvm::outs() << "[Init] Adding to worklist: " << op->getName() << "\n";
        }
    });
    llvm::outs() << "[Init] Worklist size: " << opWorklist.size() << "\n";

    int iteration = 0;

    while (!opWorklist.empty()) {
        iteration++;
        llvm::outs() << "\n--- Iteration " << iteration << " ---\n";

        auto nextop = llvm::find_if(opWorklist, [](Operation* op) {
            // bool allResolved = llvm::any_of(op->getResultTypes(), [](Type resultType) {
            //     assert(llvm::isa<mlir::cherry::CherryTensorType>(resultType));
            //     return !llvm::cast<mlir::cherry::CherryTensorType>(resultType).isDynamic();
            // });
            // return allResolved;
            bool inputsReady = llvm::all_of(op->getOperandTypes(), [](Type operandType) {
                if (auto cherryType = llvm::dyn_cast<mlir::cherry::CherryTensorType>(operandType)) {
                    return !cherryType.isDynamic();
                }
                return true;
            });

            if (!inputsReady) {
                // llvm::outs() << "  Skipping " << op->getName() << " (inputs not ready)\n";
            }
            return inputsReady;
        });
        if (nextop == opWorklist.end()) {
            llvm::outs() << "[Error] No operation is ready to be inferred!\n";
            llvm::outs() << "Remaining operations in worklist:\n";
            for (auto* op : opWorklist) {
                llvm::outs() << " - " << op->getName() << "\n";
                // 打印该 Op 的输入状态，帮助调试
                for (auto operand : op->getOperands()) {
                    llvm::outs() << "    Input type: " << operand.getType() << "\n";
                }
            }
            break;
        }
        Operation* op = *nextop;
        opWorklist.erase(op);
        llvm::outs() << "[Process] Inferring: " << op->getName() << "\n";

        if (auto shapeInferenceOp = llvm::dyn_cast<ShapeInference>(op)) {
            shapeInferenceOp.inferShapes();
            // Debug: 打印推断后的结果类型
            llvm::outs() << "  -> Result types: ";
            for (auto res : op->getResults()) {
                llvm::outs() << res.getType() << ", ";
            }
            llvm::outs() << "\n";
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
        // signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::cherry::createCherryShapeInferencePass()
{
    return std::make_unique<ShapeInferencePass>();
}
