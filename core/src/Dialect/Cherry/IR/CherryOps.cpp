#include "Dialect/Cherry/IR/CherryOps.h"

#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "Interfaces/CherryInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <string>

#define GET_OP_CLASSES
#include "Dialect/Cherry/IR/CherryOps.cpp.inc"

namespace mlir::cherry {
struct CherryInlinerInterface : public DialectInlinerInterface
{
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation* call, Operation* callable, bool wouldBeCloned) const final
    {
        return true;
    }
    bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final { return true; }
    bool isLegalToInline(Region*, Region*, bool, IRMapping&) const final { return true; }
    void handleTerminator(Operation* op, ValueRange valuesToRepl) const final
    {
        auto returnOp = cast<mlir::cherry::ReturnOp>(op);
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto& it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
    Operation* materializeCallConversion(OpBuilder& builder, Value input, Type resultType,
                                         Location conversionLoc) const final
    {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};

void CherryDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "Dialect/Cherry/IR/CherryOps.cpp.inc"
        >();
    addInterfaces<CherryInlinerInterface>();
}


//===----------------------------------------------------------------------===//
// ::mlir::cherry::CallOp
//===----------------------------------------------------------------------===//
void CallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, StringRef callee,
                   ArrayRef<mlir::Value> arguments)
{
    // Generic call always returns an unranked Tensor initially.
    state.addTypes(mlir::cherry::CherryTensorType::get(
        builder.getContext(), {mlir::ShapedType::kDynamic}, builder.getF32Type()));
    state.addOperands(arguments);
    state.addAttribute("callee", mlir::SymbolRefAttr::get(builder.getContext(), callee));
}
void CallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, StringRef callee,
                   TypeRange results, ArrayRef<mlir::Value> arguments)
{
    state.addTypes(results);
    state.addOperands(arguments);
    state.addAttribute("callee", mlir::SymbolRefAttr::get(builder.getContext(), callee));
}
CallInterfaceCallable CallOp::getCallableForCallee()
{
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}
void CallOp::setCalleeFromCallable(CallInterfaceCallable callee)
{
    (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}
Operation::operand_range CallOp::getArgOperands()
{
    return getInputs();
}
MutableOperandRange CallOp::getArgOperandsMutable()
{
    return getInputsMutable();
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::CastOp
//===----------------------------------------------------------------------===//
void CastOp::inferShapes()
{
    getResult().setType(getInput().getType());
}
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != outputs.size()) return false;
    int size = inputs.size();
    for (int i = 0; i < size; i++) {
        auto input  = llvm::dyn_cast<CherryTensorType>(inputs[i]);
        auto output = llvm::dyn_cast<CherryTensorType>(outputs[i]);
        if (!input || !output || input.getElementType() != output.getElementType()) return false;
    }
    return true;
}
struct EraseIdentityCast : public mlir::OpRewritePattern<CastOp>
{
    using OpRewritePattern<CastOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CastOp op, mlir::PatternRewriter& rewriter) const override
    {
        mlir::Value input  = op.getInput();
        mlir::Value output = op.getOutput();
        if (input.getType() == output.getType()) {
            rewriter.replaceOp(op, input);
            return mlir::success();
        }
        return mlir::failure();
    }
};

void CastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results,
                                         mlir::MLIRContext*       context)
{
    results.add<EraseIdentityCast>(context);
}
//===----------------------------------------------------------------------===//
// ::mlir::cherry::FuncOp
//===----------------------------------------------------------------------===//
void FuncOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name,
                   mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}
mlir::ParseResult FuncOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
    auto buildFuncType = [](mlir::Builder& builder,
                            llvm::ArrayRef<mlir::Type>
                                argTypes,
                            llvm::ArrayRef<mlir::Type>
                                results,
                            mlir::function_interface_impl::VariadicFlag,
                            std::string&) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(parser,
                                                          result,
                                                          /*allowVariadic=*/false,
                                                          getFunctionTypeAttrName(result.name),
                                                          buildFuncType,
                                                          getArgAttrsAttrName(result.name),
                                                          getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter& p)
{
    mlir::function_interface_impl::printFunctionOp(p,
                                                   *this,
                                                   /*isVariadic=*/false,
                                                   getFunctionTypeAttrName(),
                                                   getArgAttrsAttrName(),
                                                   getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorSliceOp
//===----------------------------------------------------------------------===//
void WeightOp::inferShapes()
{
    auto                 shape = getShapeAttr();
    int64_t              rank  = shape.size();
    SmallVector<int64_t> outputShape;
    for (auto attr : shape) {
        auto    intAttr = dyn_cast<IntegerAttr>(attr);
        int64_t size    = intAttr.getInt();
        outputShape.push_back(size);
    }
    getResult().setType(CherryTensorType::get(getContext(), outputShape, getElemType()));
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorSliceOp
//===----------------------------------------------------------------------===//
void TensorSliceOp::inferShapes()
{
    auto                 loc       = getLoc();
    Value                input     = getInput();
    auto                 inputType = cast<CherryTensorType>(input.getType());
    auto                 sizesAttr = getSizesAttr();
    int64_t              rank      = inputType.getShape().size();
    SmallVector<int64_t> outputShape;
    for (auto attr : sizesAttr) {
        auto    intAttr = dyn_cast<IntegerAttr>(attr);
        int64_t size    = intAttr.getInt();
        outputShape.push_back(size);
    }
    getResult().setType(
        CherryTensorType::get(getContext(), outputShape, inputType.getElementType()));
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorSetSliceOp
//===----------------------------------------------------------------------===//
void TensorSetSliceOp::inferShapes()
{
    getResult().setType(getDest().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::RopeOp
//===----------------------------------------------------------------------===//
void RopeOp::inferShapes()
{
    getResult().setType(getInput().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorAddop
//===----------------------------------------------------------------------===//
void TensorAddOp::inferShapes()
{
    assert(getLhs().getType() == getRhs().getType());
    getResult().setType(getLhs().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorSubOp
//===----------------------------------------------------------------------===//
void TensorSubOp::inferShapes()
{
    assert(getLhs().getType() == getRhs().getType());
    getResult().setType(getLhs().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorMulOp
//===----------------------------------------------------------------------===//
void TensorMulOp::inferShapes()
{
    assert(getLhs().getType() == getRhs().getType());
    getResult().setType(getLhs().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorDivOp
//===----------------------------------------------------------------------===//
void TensorDivOp::inferShapes()
{
    assert(getLhs().getType() == getRhs().getType());
    getResult().setType(getLhs().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorNegOp
//===----------------------------------------------------------------------===//
void TensorNegOp::inferShapes()
{
    getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorExpOp
//===----------------------------------------------------------------------===//
void TensorExpOp::inferShapes()
{
    getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorReluOp
//===----------------------------------------------------------------------===//
void TensorReluOp::inferShapes()
{
    getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorSiluOp
//===----------------------------------------------------------------------===//
void TensorSiluOp::inferShapes()
{
    getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorSigmoidOp
//===----------------------------------------------------------------------===//
void TensorSigmoidOp::inferShapes()
{
    getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorTanhOp
//===----------------------------------------------------------------------===//
void TensorTanhOp::inferShapes()
{
    getResult().setType(getOperand().getType());
}


//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorScalarArithmeticOp
//===----------------------------------------------------------------------===//
// --------------------------------------------------------------------------
// TensorAddScalarOp
// --------------------------------------------------------------------------
void TensorAddScalarOp::inferShapes()
{
    getResult().setType(getInput().getType());
}

// --------------------------------------------------------------------------
// TensorSubScalarOp
// --------------------------------------------------------------------------
void TensorSubScalarOp::inferShapes()
{
    getResult().setType(getInput().getType());
}

// --------------------------------------------------------------------------
// TensorMulScalarOp
// --------------------------------------------------------------------------
void TensorMulScalarOp::inferShapes()
{
    getResult().setType(getInput().getType());
}

// --------------------------------------------------------------------------
// TensorDivScalarOp
// --------------------------------------------------------------------------
void TensorDivScalarOp::inferShapes()
{
    getResult().setType(getInput().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::ArgmaxOp
//===----------------------------------------------------------------------===//
void ArgMaxOp::inferShapes()
{
    auto                 inputType = cast<CherryTensorType>(getInput().getType());
    auto                 rank      = inputType.getShape().size();
    SmallVector<int64_t> outputShape;
    int64_t              dim = getDim();

    for (int i = 0; i < rank; ++i) {
        if (i != dim) {
            outputShape.push_back(inputType.getShape()[i]);
        }
    }
    getResult().setType(CherryTensorType::get(
        getContext(), outputShape, cast<CherryTensorType>(getResult().getType()).getElementType()));
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::ReshapeOp
//===----------------------------------------------------------------------===//
void ReshapeOp::inferShapes()
{
    auto                          inputType   = cast<CherryTensorType>(getInput().getType());
    auto                          elementType = inputType.getElementType();
    llvm::SmallVector<int64_t, 4> newShapeDims;
    int64_t                       newShapes = 1;
    int64_t                       shapes    = 1;
    for (auto sh : inputType.getShape()) {
        shapes *= sh;
    }
    auto newShapeAttrs = getNewShapeAttr();
    for (auto newShapeAttr : newShapeAttrs) {
        auto    newShape = cast<IntegerAttr>(newShapeAttr);
        int64_t shape    = newShape.getInt();
        newShapes *= shape;
        newShapeDims.push_back(shape);
    }
    assert(shapes == newShapes);
    auto resultType = CherryTensorType::get(getContext(), newShapeDims, elementType);
    getResult().setType(resultType);
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TransposeOp
//===----------------------------------------------------------------------===//
void TransposeOp::inferShapes()
{
    auto inputType   = cast<CherryTensorType>(getInput().getType());
    auto inputShape  = inputType.getShape();
    auto elementType = inputType.getElementType();

    llvm::SmallVector<int64_t, 4> outputShape;

    ArrayAttr permAttr = getPermutation();

    for (auto attr : permAttr) {
        auto    intAttr = dyn_cast<IntegerAttr>(attr);
        int64_t index   = intAttr.getInt();
        if (index >= 0 && index < (int64_t)inputShape.size()) {
            outputShape.push_back(inputShape[index]);
        }
        else {
            outputShape.push_back(ShapedType::kDynamic);
        }
    }

    auto resultType = mlir::cherry::CherryTensorType::get(getContext(), outputShape, elementType);
    getResult().setType(resultType);
}


//===----------------------------------------------------------------------===//
// ::mlir::cherry::GenerateMaskOp
//===----------------------------------------------------------------------===//
void GenerateMaskOp::inferShapes()
{
    auto                          shapeAttr = getShapeAttr();
    llvm::SmallVector<int64_t, 4> outputShape;
    for (auto attr : shapeAttr) {
        auto intAttr = dyn_cast<IntegerAttr>(attr);
        auto dim     = intAttr.getInt();
        outputShape.push_back(dim);
    }
    auto resultType =
        CherryTensorType::get(getContext(), outputShape, FloatType::getF32(getContext()));
    getResult().setType(resultType);
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::MatMulOp
//===----------------------------------------------------------------------===//
void MaskedMatMulOp::inferShapes()
{
    auto lhsType = cast<mlir::cherry::CherryTensorType>(getLhs().getType());
    auto rhsType = cast<mlir::cherry::CherryTensorType>(getRhs().getType());

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    assert(lhsShape.size() >= 2 || rhsShape.size() >= 2);

    llvm::SmallVector<int64_t, 4> outputShape;
    size_t                        batchRank = lhsShape.size() - 2;
    for (size_t i = 0; i < batchRank; ++i) {
        outputShape.push_back(lhsShape[i]);
    }
    outputShape.push_back(lhsShape[lhsShape.size() - 2]);
    outputShape.push_back(rhsShape[rhsShape.size() - 1]);
    auto resultType =
        mlir::cherry::CherryTensorType::get(getContext(), outputShape, lhsType.getElementType());
    getResult().setType(resultType);
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::MatMulOp
//===----------------------------------------------------------------------===//
void MatMulOp::inferShapes()
{
    auto lhsType = cast<mlir::cherry::CherryTensorType>(getLhs().getType());
    auto rhsType = cast<mlir::cherry::CherryTensorType>(getRhs().getType());

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    assert(lhsShape.size() >= 2 || rhsShape.size() >= 2);

    llvm::SmallVector<int64_t, 4> outputShape;
    size_t                        batchRank = lhsShape.size() - 2;
    for (size_t i = 0; i < batchRank; ++i) {
        outputShape.push_back(lhsShape[i]);
    }
    outputShape.push_back(lhsShape[lhsShape.size() - 2]);
    outputShape.push_back(rhsShape[rhsShape.size() - 1]);
    auto resultType =
        mlir::cherry::CherryTensorType::get(getContext(), outputShape, lhsType.getElementType());
    getResult().setType(resultType);
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::SoftmaxOp
//===----------------------------------------------------------------------===//
void SoftmaxOp::inferShapes()
{
    getResult().setType(getInput().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::LayerNormOp
//===----------------------------------------------------------------------===//
void LayerNormOp::inferShapes()
{
    getResult().setType(getInput().getType());
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::BroadcastOp
//===----------------------------------------------------------------------===//
void BroadcastOp::inferShapes()
{
    auto       inputType   = cast<mlir::cherry::CherryTensorType>(getInput().getType());
    mlir::Type elementType = inputType.getElementType();

    // TODO: now only for [1] -> [target]
    assert(inputType.getShape().size() == 1);

    llvm::SmallVector<int64_t, 4> targetShapeDims;

    for (auto dimVal : getTargetShape()) {
        if (auto constantOp = dimVal.getDefiningOp<ConstantOp>()) {
            mlir::Attribute attr = constantOp.getValue();
            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
                int64_t dim = intAttr.getInt();
                targetShapeDims.push_back(dim);
            }
            else {
                targetShapeDims.push_back(mlir::ShapedType::kDynamic);
            }
        }
        else {
            targetShapeDims.push_back(mlir::ShapedType::kDynamic);
        }
    }

    auto resultType =
        mlir::cherry::CherryTensorType::get(getContext(), targetShapeDims, elementType);
    getResult().setType(resultType);
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::RMSNormOp
//===----------------------------------------------------------------------===//
void RMSNormOp::inferShapes()
{
    getResult().setType(getInput().getType());
}
}   // namespace mlir::cherry
