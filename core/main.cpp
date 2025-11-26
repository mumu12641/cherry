#include "dialect/cherry/IR/CherryDialect.h"
#include "dialect/cherry/IR/CherryTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

int main()
{
    mlir::DialectRegistry registry;
    mlir::MLIRContext     context(registry);
    auto                  dialect = context.getOrLoadDialect<mlir::cherry::CherryDialect>();
    dialect->registerType();

    mlir::cherry::CherryTensorType ch_tensor = mlir::cherry::CherryTensorType::get(
        &context, {1, 2, 3}, mlir::Float32Type::get(&context));
    llvm::outs() << " Tensor 类型 :\t";
    ch_tensor.dump();
    // mlir::cherry::CherryTensorType dy_tensor = mlir::cherry::CherryTensorType::get(
    //     &context, {mlir::ShapedType::kDynamic, 2, 3}, mlir::Float32Type::get(&context), 3);
    // llvm::outs() << "动态  Tensor 类型 :\t";
    // dy_tensor.dump();
}