#include "dialect/cherry/IR/CherryOps.h"

#include "dialect/cherry/IR/CherryDialect.h"
#include "dialect/cherry/IR/CherryTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

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
    
    auto* inlinerInterface =
        this->getRegisteredInterface<mlir::cherry::CherryInlinerInterface>();

    if (inlinerInterface) {
        llvm::errs() << "✅ Success: CherryInlinerInterface is registered correctly!\n";
    }
    else {
        llvm::errs() << "❌ Failure: CherryInlinerInterface is NOT registered.\n";
        llvm::errs() << "   Please check CherryDialect::initialize() in CherryDialect.cpp\n";
    }
}


//===----------------------------------------------------------------------===//
// ::mlir::cherry::CallOp
//===----------------------------------------------------------------------===//
void CallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, StringRef callee,
                   ArrayRef<mlir::Value> arguments)
{
    // Generic call always returns an unranked Tensor initially.
    state.addTypes(mlir::cherry::CherryTensorType::get(
        builder.getContext(),
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, 8},
        builder.getF32Type()));
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
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1) return false;
    return true;
    // The inputs must be Tensors with the same element type.
    // TensorType input  = llvm::dyn_cast<TensorType>(inputs.front());
    // TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
    // if (!input || !output || input.getElementType() != output.getElementType()) return false;
    // // The shape is required to match if both types are ranked.
    // return !input.hasRank() || !output.hasRank() || input == output;
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
}   // namespace mlir::cherry