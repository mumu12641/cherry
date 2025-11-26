#include "dialect/cherry/IR/CherryOps.h"
#include "dialect/cherry/IR/CherryTypes.h"
#include "dialect/cherry/IR/CherryDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "dialect/cherry/IR/CherryOps.cpp.inc"

namespace mlir::cherry {
void CherryDialect::registerOps()
{
    llvm::outs() << "register " << getDialectNamespace() << "  Ops\n";
    addOperations<
#define GET_OP_LIST
#include "dialect/cherry/IR/CherryOps.cpp.inc"
        >();
}
}   // namespace mlir::cherry