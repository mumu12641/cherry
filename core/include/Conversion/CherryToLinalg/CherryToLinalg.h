#ifndef DIALECT_CHERRY_TO_LINALG_H
#define DIALECT_CHERRY_TO_LINALG_H

#include "mlir/Pass/Pass.h"

#include <memory>
namespace mlir {
class TypeConverter;
}

namespace mlir::cherry {

#define GEN_PASS_DECL_CONVERTCHERRYTOLINALGPASS
#include "Conversion/Passes.h.inc"

}   // namespace mlir::cherry

#endif
