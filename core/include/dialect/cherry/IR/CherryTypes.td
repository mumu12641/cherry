#ifndef DIALECT_CHERRY_TYPES_TD
#define DIALECT_CHERRY_TYPES_TD

include "dialect/cherry/IR/CherryDialect.td"
// include "CherryDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

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

// def Cherry_AnyCherryTensor : Type<Cherry_CherryTensorType.predicate, "any cherry tensor">;

#endif
