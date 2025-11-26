#ifndef DIALECT_CHERRY_OPS_TD
#define DIALECT_CHERRY_OPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "dialect/cherry/IR/CherryDialect.td"
include "dialect/cherry/IR/CherryTypes.td"

class Cherry_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cherry_Dialect, mnemonic, traits> {
  let summary = cppNamespace#opName#"op";
  let description = "$_name op";
}

def Cherry_ConstantOp : Cherry_Op<"constant", [NoSideEffect]>{

}

#endif