#ifndef DIALECT_CHERRY_OPS_TD
#define DIALECT_CHERRY_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/Interfaces/InferTypeOpInterface.td"
include "dialect/cherry/IR/CherryTypes.td"
// include "CherryTypes.td"

class Cherry_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cherry_Dialect, mnemonic, traits> {
  let summary = "Cherry operation";
  let description = [{
    Base class for operations in the Cherry dialect.
  }];
}

// ===----------------------------------------------------------------------===//
// ConstantOp
// ===----------------------------------------------------------------------===//
def Cherry_ConstantOp : Cherry_Op<"constant", [Pure]> {
  let summary = "Constant scalar operation";
  let description = [{
    Creates a constant value.
    Example:
      %0 = cherry.constant 100.0 : f32
      %1 = cherry.constant 42 : i32
  }];

  let arguments = (ins AnyAttr:$value);
  
  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
}

// ===----------------------------------------------------------------------===//
// AddOp
// ===----------------------------------------------------------------------===//
def Cherry_AddOp : Cherry_Op<"add", [Pure, Commutative]> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition of two CherryTensors.
    The operands and result must have the same shape and element type.
    
    Example:
      %2 = cherry.add %0, %1 : !cherry.cherry_tensor<[2, 2], f32>
  }];

  let arguments = (ins AnyCherryTensor:$lhs, AnyCherryTensor:$rhs);
  
  let results = (outs AnyCherryTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

#endif
