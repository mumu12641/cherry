#ifndef DIALECT_CHERRY_TYPES_TD
#define DIALECT_CHERRY_TYPES_TD

include "dialect/cherry/IR/CherryDialect.td"
// include "CherryDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

class Cherry_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Cherry_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = dialect.name # "." # typeMnemonic;
}

def Cherry_CherryTensorType : Cherry_Type<"CherryTensor","cherry_tensor",[]>{
    let summary = " the summary of cherry tensor type";

    let description = "description of cherry tensor type";

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "Type":$elementType
    );

    let hasCustomAssemblyFormat = 1;
}

def AnyCherryTensor : Type<And<[
    CPred<"::mlir::isa<::mlir::cherry::CherryTensorType>($_self)">,
    CPred<"::mlir::cast<::mlir::cherry::CherryTensorType>($_self).getShape().size() > 0">
]>, "any cherry tensor with rank > 0">;

// 定义一个 Integer 类型约束
def AnyIntegerType : Type<
    CPred<"::llvm::isa<::mlir::IntegerType>($_self)">, 
    "integer"
>;

def AnyCherryScalar : AnyTypeOf<[F32, F64, AnyIntegerType], "f32, f64, or integer">;

// def AnyCherryScalar : Type<Or<[
//     F32, 
//     F64, 
//     CPred<"::llvm::isa<::mlir::IntegerType>($_self)">
// ]>, "f32, f64, or integer">;

#endif
