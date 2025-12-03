#ifndef DIALECT_CHERRY_PASSES_H
#define DIALECT_CHERRY_PASSES_H

#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "mlir/Pass/Pass.h"
namespace mlir {
namespace cherry {

std::unique_ptr<Pass> createCherryShapeInferencePass();
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Dialect/Cherry/Transforms/Passes.h.inc"

}   // namespace cherry
}   // namespace mlir
#endif