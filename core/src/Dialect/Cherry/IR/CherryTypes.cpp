#include "Dialect/Cherry/IR/CherryTypes.h"

#include "Dialect/Cherry/IR/CherryDialect.h"
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
#include "Dialect/Cherry/IR/CherryTypes.cpp.inc"


namespace mlir::cherry {
void CherryDialect::registerType()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Cherry/IR/CherryTypes.cpp.inc"
        >();
}

bool CherryTensorType::isDynamic()
{
    for (auto dim : getShape())
        if (dim < 0) return true;
    return false;
}

void CherryTensorType::print(::mlir::AsmPrinter& printer) const
{
    printer << "<";
    printer << "[";
    for (auto dim : getShape()) {
        if (dim < 0) {
            printer << "?" << "x";
        }
        else {
            printer << dim << "x";
        }
    }
    printer.printType(getElementType());
    printer << "]";
    printer << ">";
}

Type CherryTensorType::parse(::mlir::AsmParser& parser)
{
    if (parser.parseLess()) return Type();
    if (parser.parseLSquare()) return Type();
    SmallVector<int64_t> shape;
    if (parser.parseDimensionList(shape, /*allowDynamic=*/true, /*withTrailingX=*/true)) {
        return Type();
    }
    Type elementType;
    if (parser.parseType(elementType)) return Type();
    if (parser.parseRSquare()) return Type();
    if (parser.parseGreater()) return Type();
    return get(parser.getContext(), shape, elementType);
}


}   // namespace mlir::cherry
