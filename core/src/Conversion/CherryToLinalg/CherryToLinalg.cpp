#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
// #include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "Conversion/CherryToLinalg/CherryToLinalg.h"
#include "Dialect/Cherry/IR/CherryDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>


namespace mlir::cherry {
#define GEN_PASS_DEF_CONVERTCHERRYTOLINALGPASS
#include "Conversion/Passes.h.inc"
}   // namespace mlir::cherry

struct ConvertCherryToLinalgPass
    : mlir::cherry::impl::ConvertCherryToLinalgPassBase<ConvertCherryToLinalgPass>
{
    using mlir::cherry::impl::ConvertCherryToLinalgPassBase<
        ConvertCherryToLinalgPass>::ConvertCherryToLinalgPassBase;
    void runOnOperation() override;
};

void ConvertCherryToLinalgPass::runOnOperation()
{
    // Implementation of the pass
    llvm::outs() << "Running Cherry to Linalg conversion pass\n";
}

