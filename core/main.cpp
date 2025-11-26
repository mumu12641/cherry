#include "dialect/cherry/IR/CherryDialect.h"
#include "dialect/cherry/IR/CherryOps.h"
#include "dialect/cherry/IR/CherryTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/raw_ostream.h"

int main()
{
    mlir::DialectRegistry registry;
    mlir::MLIRContext     context(registry);
    context.loadAllAvailableDialects();
    auto dialect = context.getOrLoadDialect<mlir::cherry::CherryDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    dialect->registerType();
    dialect->registerOps();

    mlir::OpBuilder                   builder(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    // 定义类型: 2x3 的 f32 Tensor
    auto f32Type    = builder.getF32Type();
    auto tensorType = mlir::cherry::CherryTensorType::get(&context, {2, 3}, f32Type);
    llvm::outs() << " Tensor 类型 :\t";
    mlir::cherry::CherryTensorType dy_tensor = mlir::cherry::CherryTensorType::get(
        &context, {mlir::ShapedType::kDynamic, 2, 3}, mlir::Float32Type::get(&context));
    llvm::outs() << "动态  Tensor 类型 :\t";
    dy_tensor.dump();
    tensorType.dump();


    llvm::outs() << "Creating Operations...\n";

    auto floatAttr = builder.getF32FloatAttr(1.5f);
    auto constOp1  = builder.create<mlir::cherry::ConstantOp>(
        builder.getUnknownLoc(), mlir::Float32Type::get(&context), floatAttr);

    auto funcType = mlir::FunctionType::get(&context, {tensorType, tensorType}, {tensorType});
    auto funcOp =
        builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "test_add_func", funcType);
    mlir::Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    mlir::Value lhs = entryBlock->getArgument(0);
    mlir::Value rhs = entryBlock->getArgument(1);
    auto addOp = builder.create<mlir::cherry::AddOp>(builder.getUnknownLoc(), tensorType, lhs, rhs);
    addOp.dump();
    // builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{addOp});
    module->print(llvm::outs());

    return 0;
}