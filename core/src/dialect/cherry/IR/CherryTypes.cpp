#include "dialect/cherry/IR/CherryTypes.h"

#include "dialect/cherry/IR/CherryDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#define FIX
#define GET_TYPEDEF_CLASSES
#include "dialect/cherry/IR/CherryTypes.cpp.inc"


namespace mlir::cherry {
void CherryDialect::registerType()
{
    llvm::outs() << "register " << getDialectNamespace() << "  Type\n";
    addTypes<
#define GET_TYPEDEF_LIST
#include "dialect/cherry/IR/CherryTypes.cpp.inc"
        >();
}

void CherryTensorType::print(::mlir::AsmPrinter& printer) const
{
    printer << "<";
    printer << "[";
    llvm::interleaveComma(getShape(), printer);
    printer << "]";
    printer << ", ";
    printer.printType(getElementType());
    printer << ">";
}

Type CherryTensorType::parse(::mlir::AsmParser& parser)
{
    if (parser.parseLess()) return {};
    llvm::SmallVector<int64_t> shape;
    auto                       parseInt = [&]() -> ::mlir::ParseResult {
        int64_t val;
        if (parser.parseInteger(val)) return ::mlir::failure();
        shape.push_back(val);
        return ::mlir::success();
    };
    if (parser.parseCommaSeparatedList(::mlir::AsmParser::Delimiter::Square, parseInt)) {
        return {};
    }
    if (parser.parseComma()) return {};
    Type elementType;
    if (parser.parseType(elementType)) return {};
    if (parser.parseGreater()) return {};
    return parser.getChecked<CherryTensorType>(parser.getContext(), shape, elementType);
}


}   // namespace mlir::cherry