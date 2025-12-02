#include "dialect/cherry/IR/CherryOps.h"

#include "dialect/cherry/IR/CherryDialect.h"
#include "dialect/cherry/IR/CherryTypes.h"
#include "interfaces/CherryInterface.h"
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
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

#define GET_OP_CLASSES
#include "dialect/cherry/IR/CherryOps.cpp.inc"

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
    llvm::outs() << "register " << getDialectNamespace() << "  Ops\n";
    addOperations<
#define GET_OP_LIST
#include "dialect/cherry/IR/CherryOps.cpp.inc"
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
// void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1) return false;
    auto input  = llvm::dyn_cast<CherryTensorType>(inputs.front());
    auto output = llvm::dyn_cast<CherryTensorType>(outputs.front());
    if (!input || !output || input.getElementType() != output.getElementType()) return false;
    return true;
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
// ::mlir::cherry::ReshapeOp
//===----------------------------------------------------------------------===//
void ReshapeOp::inferShapes()
{
    auto                          inputType   = cast<CherryTensorType>(getInput().getType());
    auto                          elementType = inputType.getElementType();
    llvm::SmallVector<int64_t, 4> newShapeDims;
    for (auto shapeDim : getNewShape()) {
        llvm::APInt dimValue;
        if (mlir::matchPattern(shapeDim, mlir::m_ConstantInt(&dimValue))) {
            int64_t dim = dimValue.getSExtValue();
            newShapeDims.push_back(dim);
        }
        else {
            newShapeDims.push_back(mlir::ShapedType::kDynamic);
        }
    }
    auto resultType = CherryTensorType::get(getContext(), newShapeDims, elementType);
    getResult().setType(resultType);
}

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TransposeOp
//===----------------------------------------------------------------------===//
void TransposeOp::inferShapes()
{
    auto                          inputType   = cast<CherryTensorType>(getInput().getType());
    auto                          inputShape  = inputType.getShape();
    auto                          elementType = inputType.getElementType();
    llvm::SmallVector<int64_t, 4> outputShape;

    for (mlir::Value permVal : getPermutation()) {
        llvm::APInt permIndex;
        int64_t     idx = permIndex.getSExtValue();
        assert(idx >= 0 && idx < (int64_t)inputShape.size());
        outputShape.push_back(inputShape[idx]);
    }

    auto resultType = mlir::cherry::CherryTensorType::get(getContext(), outputShape, elementType);
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

    assert(lhsShape.size() < 2 || rhsShape.size() < 2);

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

    llvm::SmallVector<int64_t, 4> targetShapeDims;

    for (mlir::Value dimVal : getTargetShape()) {
        llvm::APInt dimConst;
        if (mlir::matchPattern(dimVal, mlir::m_ConstantInt(&dimConst))) {
            targetShapeDims.push_back(dimConst.getSExtValue());
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
// ::mlir::cherry::CastOp
//===----------------------------------------------------------------------===//
void CastOp::inferShapes()
{
    getResult().setType(getInput().getType());
}
}   // namespace mlir::cherry