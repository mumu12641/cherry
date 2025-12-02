#ifndef DIALECT_CHERRY_INERFACE_TD
#define DIALECT_CHERRY_INERFACE_TD

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif

