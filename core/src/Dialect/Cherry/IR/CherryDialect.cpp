
#include "Dialect/Cherry/IR/CherryDialect.h"

#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/Support/raw_ostream.h"

#define FIX
#include "Dialect/Cherry/IR/CherryDialect.cpp.inc"
#undef FIX

namespace mlir::cherry {
void CherryDialect::initialize()
{
    llvm::outs() << "initializing " << getDialectNamespace() << "\n";
    registerType();
    registerOps();
}

CherryDialect::~CherryDialect()
{
    llvm::outs() << "destroying " << getDialectNamespace() << "\n";
}
}   // namespace mlir::cherry
