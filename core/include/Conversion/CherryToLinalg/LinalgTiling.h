#ifndef DIALECT_LINALG_TILING_H
#define DIALECT_LINALG_TILING_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::cherry {
#define GEN_PASS_DECL_CHERRYLINALGTILINGPASS
#include "Conversion/Passes.h.inc"

}   // namespace mlir::cherry

#endif
