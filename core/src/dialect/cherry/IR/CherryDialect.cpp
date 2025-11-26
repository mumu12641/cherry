
#include "dialect/cherry/IR/CherryDialect.h"

#include "llvm/Support/raw_ostream.h"

// #define FIX
#include "dialect/cherry/IR/CherryDialect.cpp.inc"
// #undef FIX

namespace mlir::cherry {
void CherryDialect::initialize()
{
    llvm::outs() << "initializing12 " << getDialectNamespace() << "\n";
}

CherryDialect::~CherryDialect()
{
    llvm::outs() << "destroying " << getDialectNamespace() << "\n";
}
}   // namespace mlir::cherry
