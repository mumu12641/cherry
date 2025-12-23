#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "Dialect/Cherry/Transforms/Passes.h"
#include "Interfaces/CherryInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/RDFGraph.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
namespace mlir::cherry {
#define GEN_PASS_DEF_SHAPEINFERENCEPASS
#include "Dialect/Cherry/Transforms/Passes.h.inc"
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
        else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
            auto   initArgs = forOp.getInitArgs();
            auto   results  = forOp.getResults();
            Block* body     = forOp.getBody();
            for (size_t i = 0; i < initArgs.size(); ++i) {
                Type staticType = initArgs[i].getType();
                results[i].setType(staticType);
                body->getArgument(i + 1).setType(staticType);
            }
        }
        else if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(op)) {
            auto   inits       = whileOp.getInits();
            Block& beforeBlock = whileOp.getBefore().front();
            Block& afterBlock  = whileOp.getAfter().front();

            for (size_t i = 0; i < inits.size(); ++i) {
                Type staticType = inits[i].getType();
                beforeBlock.getArgument(i).setType(staticType);
            }

            auto conditionOp   = llvm::dyn_cast<scf::ConditionOp>(beforeBlock.getTerminator());
            auto conditionArgs = conditionOp.getArgs();
            auto results       = whileOp.getResults();

            for (size_t i = 0; i < conditionArgs.size(); ++i) {
                Type staticType = conditionArgs[i].getType();

                results[i].setType(staticType);

                afterBlock.getArgument(i).setType(staticType);
            }
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

    funcOp.walk([&](Operation* op) {
        if (auto returnOp = llvm::dyn_cast<ReturnOp>(op)) {
            auto returnType      = returnOp.getOperandTypes();
            auto currentFuncType = funcOp.getResultTypes();
            auto newFuncType     = FunctionType::get(
                funcOp.getContext(), funcOp.getFunctionType().getInputs(), TypeRange{returnType});
            funcOp.setType(newFuncType);
        }
    });
}

std::unique_ptr<mlir::Pass> mlir::cherry::createCherryShapeInferencePass()
{
    return std::make_unique<ShapeInferencePass>();
}
