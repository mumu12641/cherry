#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
// #include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "Conversion/CherryToLinalg/CherryToLinalg.h"
#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::cherry {
#define GEN_PASS_DEF_CONVERTCHERRYTOLINALGPASS
#include "Conversion/Passes.h.inc"
}   // namespace mlir::cherry

using namespace mlir;
using namespace mlir::cherry;
class CherryTypeConverter : public TypeConverter
{
public:
    CherryTypeConverter()
    {
        addConversion([](Type type) { return type; });
        addConversion([](CherryTensorType type) -> Type {
            return RankedTensorType::get(type.getShape(), type.getElementType());
        });
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::ConstantOp lowering
// ===----------------------------------------------------------------------===//
struct ConstantOpLowering : public OpConversionPattern<ConstantOp>
{
    using OpConversionPattern<ConstantOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                       llvm::cast<mlir::TypedAttr>(op.getValue()));
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::CreateTensorOp lowering
// ===----------------------------------------------------------------------===//
struct CreateTensorOpLowering : public OpConversionPattern<CreateTensorOp>
{
    using OpConversionPattern<CreateTensorOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CreateTensorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto valueAttr  = op.getValue();
        auto targetType = typeConverter->convertType(op.getResult().getType());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, targetType, valueAttr);
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::ReturnOp Lowering
// ===----------------------------------------------------------------------===//
struct ReturnOpLowering : public OpConversionPattern<ReturnOp>
{
    using OpConversionPattern<ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::FuncOp Lowering
// ===----------------------------------------------------------------------===//
struct FuncOpLowering : public OpConversionPattern<FuncOp>
{
    using OpConversionPattern<FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto                               type = op.getFunctionType();
        TypeConverter::SignatureConversion result(type.getNumInputs());

        for (auto [index, inputType] : llvm::enumerate(type.getInputs())) {
            if (failed(typeConverter->convertSignatureArg(index, inputType, result)))
                return failure();
        }

        SmallVector<Type, 1> newResultTypes;
        if (failed(typeConverter->convertTypes(type.getResults(), newResultTypes)))
            return failure();

        auto newFuncOp = rewriter.create<func::FuncOp>(
            op.getLoc(),
            op.getSymName(),
            rewriter.getFunctionType(result.getConvertedTypes(), newResultTypes));

        rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.end());

        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter, &result)))
            return failure();

        rewriter.eraseOp(op);
        return success();
    }
};

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
    ModuleOp            module = getOperation();
    CherryTypeConverter converter;

    ConversionTarget target(getContext());

    target.addLegalDialect<linalg::LinalgDialect,
                           func::FuncDialect,
                           tensor::TensorDialect,
                           CherryDialect,
                           arith::ArithDialect>();

    // // Cherry Dialect 是非法的 (必须被转掉)
    // target.addIllegalDialect<CherryDialect>();

    // 也可以设置具体的 Op 为非法
    target.addIllegalOp<cherry::ConstantOp>();
    target.addIllegalOp<cherry::CreateTensorOp>();
    target.addIllegalOp<cherry::ReturnOp>();
    target.addIllegalOp<cherry::FuncOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, CreateTensorOpLowering, ReturnOpLowering, FuncOpLowering>(
        converter, &getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
