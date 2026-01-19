#ifndef DIALECT_INSERT_HOST_REGISTER_H
#define DIALECT_INSERT_HOST_REGISTER_H

#include "mlir/Pass/Pass.h"

namespace mlir::cherry {

#define GEN_PASS_DECL_CHERRYINSERTHOSTREGISTERPASS
#include "Conversion/Passes.h.inc"

}   // namespace mlir::cherry

#endif
