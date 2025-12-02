#ifndef DIALECT_CHERRY_PASSES_H
#define DIALECT_CHERRY_PASSES_H

#include "dialect/cherry/IR/CherryDialect.h"
#include "mlir/Pass/Pass.h"
namespace mlir {
namespace cherry {

std::unique_ptr<Pass> createCherryShapeInferencePass();
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "dialect/cherry/Transforms/Passes.h.inc"

}   // namespace cherry
}   // namespace mlir
#endif