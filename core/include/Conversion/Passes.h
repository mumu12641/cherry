#ifndef DIALECT_CHERRY_CONVERSION_PASS_H
#define DIALECT_CHERRY_CONVERSION_PASS_H

#include "mlir/Pass/Pass.h"
namespace mlir::cherry {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
}   // namespace mlir::cherry
#endif

