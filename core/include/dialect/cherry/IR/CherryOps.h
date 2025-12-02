#ifndef DIALECT_CHERRY_OPS_H
#define DIALECT_CHERRY_OPS_H

// #include "dialect/cherry/IR/CherryDialect.h"
// #include "dialect/cherry/IR/CherryTypes.h"
#include "interfaces/CherryInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define FIX
#define GET_OP_CLASSES
#include "dialect/cherry/IR/CherryOps.h.inc"
#undef FIX

#endif   // DIALECT_CHERRY_OPS_H
