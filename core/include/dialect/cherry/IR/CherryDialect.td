#ifndef DIALECT_CHERRY
#define DIALECT_CHERRY
include "mlir/IR/DialectBase.td"

def Cherry_Dialect : Dialect{
  let name = "cherry";

  let summary = "summary of Cherry Dialect";

  let description = "description of Cherry Dialect";

  let dependentDialects = ["::mlir::tensor::TensorDialect"];

  let discardableAttrs = (ins);

  let cppNamespace = "::mlir::cherry";

  let extraClassDeclaration = [{
      void registerType();
      // void registerOps();
  }];
  
  let hasConstantMaterializer = 0;

  let hasNonDefaultDestructor = 1;

  let hasOperationAttrVerify = 0;

  let hasRegionArgAttrVerify = 0;

  let hasRegionResultAttrVerify = 0;

  let hasOperationInterfaceFallback = 0;

  let useDefaultAttributePrinterParser = 0;

  let useDefaultTypePrinterParser = 1;

  let hasCanonicalizer = 0;

  let isExtensible = 0;

  let usePropertiesForAttributes = 1;

}

#endif // DIALECT_NORTH_STAR
