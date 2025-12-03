#ifndef DIALECT_CHERRY_TYPES_H
#define DIALECT_CHERRY_TYPES_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"

#define FIX
#define GET_TYPEDEF_CLASSES
#include "Dialect/Cherry/IR/CherryTypes.h.inc"
#undef FIX



#endif   // DIALECT_CHERRY_TYPES_H
