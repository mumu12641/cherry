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
  let results = (outs AnyCherryScalar:$result);
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
}

def Cherry_CreateTensorOp : Cherry_Op<"create_tensor", [Pure]> {
  let summary = "Create a constant tensor from attributes";
  let description = [{
    Creates a CherryTensor from a dense elements attribute.
    Example: 
      %t = cherry.create_tensor dense<[1.0, 2.0]> : tensor<2xf32> -> !cherry.tensor<[2], f32>
  }];

  let arguments = (ins ElementsAttr:$value);
  
  let results = (outs AnyCherryTensor:$result);

  let assemblyFormat = "$value attr-dict `->` type($result)";
}


// ===----------------------------------------------------------------------===//
// Scalar Binary Operations (标量二元运算)
// ===----------------------------------------------------------------------===//

class Cherry_ScalarBinaryOp<string mnemonic, list<Trait> traits = []> 
    : Cherry_Op<mnemonic, !listconcat([Pure], traits)> {
  
  let arguments = (ins AnyCherryScalar:$lhs, AnyCherryScalar:$rhs);
  let results = (outs AnyCherryScalar:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Cherry_ScalarAddOp : Cherry_ScalarBinaryOp<"scalar_add", [Commutative]> {
  let summary = "Scalar addition";
}

def Cherry_ScalarSubOp : Cherry_ScalarBinaryOp<"scalar_sub"> {
  let summary = "Scalar subtraction";
}

def Cherry_ScalarMulOp : Cherry_ScalarBinaryOp<"scalar_mul", [Commutative]> {
  let summary = "Scalar multiplication";
}

def Cherry_ScalarDivOp : Cherry_ScalarBinaryOp<"scalar_div"> {
  let summary = "Scalar division";
}


class Cherry_BinaryTensorOp<string mnemonic, list<Trait> traits = []> 
    : Cherry_Op<mnemonic, !listconcat([Pure], traits)> { // 默认都带 Pure
  
  let arguments = (ins AnyCherryTensor:$lhs, AnyCherryTensor:$rhs);
  let results = (outs AnyCherryTensor:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

class Cherry_UnaryTensorOp<string mnemonic, list<Trait> traits = []> 
    : Cherry_Op<mnemonic, !listconcat([Pure], traits)> {
  
  let arguments = (ins AnyCherryTensor:$operand);
  let results = (outs AnyCherryTensor:$result);
  
  let assemblyFormat = "$operand attr-dict `:` functional-type(operands, results)";
}

// ===----------------------------------------------------------------------===//
// Tensor Binary Operations 
// ===----------------------------------------------------------------------===//

def Cherry_TensorAddOp : Cherry_BinaryTensorOp<"tensor_add", [Commutative]> {
  let summary = "Element-wise addition";
  let description = [{
    Performs element-wise addition of two CherryTensors.
    Example: %z = cherry.add %x, %y : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  }];
}

def Cherry_TensorSubOp : Cherry_BinaryTensorOp<"tensor_sub"> {
  let summary = "Element-wise subtraction";
  let description = "Performs element-wise subtraction: lhs - rhs";
}

def Cherry_TensorMulOp : Cherry_BinaryTensorOp<"tensor_mul">{
  let summary = "Element-wise multiplication";
  let description = "Performs element-wise multiplication: lhs * rhs";
}

def Cherry_TensorDivOp : Cherry_BinaryTensorOp<"tensor_div">{
  let summary = "Element-wise division";
  let description = "Performs element-wise division: lhs / rhs";
}

// ===----------------------------------------------------------------------===//
// Tensor Unary Operations 
// ===----------------------------------------------------------------------===//

def Cherry_TensorNegOp : Cherry_UnaryTensorOp<"tensor_neg"> {
  let summary = "Element-wise negation";
}

def Cherry_TensorExpOp : Cherry_UnaryTensorOp<"tensor_exp"> {
  let summary = "Element-wise exponential";
}

def Cherry_TensorReluOp : Cherry_UnaryTensorOp<"tensor_relu"> {
  let summary = "Element-wise relu";
}
def  Cherry_TensorSigmoidOp : Cherry_UnaryTensorOp<"tensor_sigmoid"> {
  let summary = "Element-wise sigmoid";
}

def  Cherry_TensorTanhOp : Cherry_UnaryTensorOp<"tensor_tanh"> {
  let summary = "Element-wise tanh";
}


//===----------------------------------------------------------------------===//
// Tensor Reduction Operations 
//===----------------------------------------------------------------------===//

class Cherry_ReductionOp<string mnemonic> : Cherry_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyCherryTensor:$input, Optional<I64>:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input (`,` $axis^)? attr-dict `:` type($input) `->` type($result)";
}

// Sum
def Cherry_SumOp : Cherry_ReductionOp<"sum"> {
  let summary = "Tensor sum reduction";
  let description = [{
    Reduces a tensor by summing all elements (full reduction) or along a specific axis.
    Example:
      %sum = cherry.sum %tensor : !cherry.tensor<2x3xf32> -> f32
  }];
}

// Mean
def Cherry_MeanOp : Cherry_ReductionOp<"mean"> {
  let summary = "Tensor mean reduction";
  let description = [{
    Reduces a tensor by computing the mean of all elements.
    Example:
      %mean = cherry.mean %tensor : !cherry.tensor<2x3xf32> -> f32
  }];
}

// Max
def Cherry_MaxReduceOp : Cherry_ReductionOp<"max_reduce"> { 
  let summary = "Tensor max reduction";
  let description = [{
    Reduces a tensor by finding the maximum element.
    Example:
      %max = cherry.max_reduce %tensor : !cherry.tensor<2x3xf32> -> f32
  }];
}

// Min
def Cherry_MinReduceOp : Cherry_ReductionOp<"min_reduce"> { 
  let summary = "Tensor min reduction";
  let description = [{
    Reduces a tensor by finding the minimum element.
    Example:
      %min = cherry.min_reduce %tensor : !cherry.tensor<2x3xf32> -> f32
  }];
}

// Argmax
def Cherry_ArgmaxOp : Cherry_ReductionOp<"argmax"> {
  let summary = "Tensor argmax reduction";
  let description = [{
    Reduces a tensor by finding the index of the maximum element.
    Example:
      %idx = cherry.argmax %tensor : !cherry.tensor<2x3xf32> -> i64
  }];
}

//===----------------------------------------------------------------------===//
// Tensor Memory/Shape Operations 
//===----------------------------------------------------------------------===//

// Reshape
def Cherry_ReshapeOp : Cherry_Op<"reshape", [Pure]> {
  let summary = "Reshape tensor without copying data";
  let description = [{
    Changes the shape of a tensor without reordering elements. Total number of elements must remain the same.
    Example:
      %reshaped = cherry.reshape %input, %d0, %d1 : (!cherry.tensor<6xf32>, i64, i64) -> !cherry.tensor<2x3xf32>
  }];
  
  let arguments = (ins AnyCherryTensor:$input, Variadic<I64>:$new_shape);
  let results = (outs AnyCherryTensor:$result);
  
  let assemblyFormat = "$input `,` $new_shape attr-dict `:` `(` type($input) `,` type($new_shape) `)` `->` type($result)";
}

def Cherry_TransposeOp : Cherry_Op<"transpose", [Pure]> {
  let summary = "Transpose tensor dimensions";
  let description = [{
    Swaps tensor dimensions according to a permutation.
    Example:
      %transposed = cherry.transpose %input, %p0, %p1 : (!cherry.tensor<2x3xf32>, i64, i64) -> !cherry.tensor<3x2xf32>
  }];
  
  let arguments = (ins AnyCherryTensor:$input, Variadic<I64>:$permutation);
  let results = (outs AnyCherryTensor:$result);

  let assemblyFormat = "$input (`,` $permutation^)? attr-dict `:` `(` type($input) (`,` type($permutation)^)? `)` `->` type($result)";
}

// ===----------------------------------------------------------------------===//
// Essential Operations
// ===----------------------------------------------------------------------===//

// 1. MatMul: 支持 Batch 的矩阵乘法
// [B, M, K] * [B, K, N] -> [B, M, N]
def Cherry_MatMulOp : Cherry_Op<"matmul", [Pure]> {
  let summary = "Batch matrix multiplication";
  let arguments = (ins AnyCherryTensor:$lhs, AnyCherryTensor:$rhs);
  let results = (outs AnyCherryTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

// 2. Softmax: 指定轴进行 Softmax
def Cherry_SoftmaxOp : Cherry_Op<"softmax", [Pure]> {
  let summary = "Softmax operation";
  let arguments = (ins AnyCherryTensor:$input, I64Attr:$axis);
  let results = (outs AnyCherryTensor:$result);
  let assemblyFormat = "$input `axis` $axis attr-dict `:` functional-type(operands, results)";
}

// 3. LayerNorm: 层归一化
// y = (x - mean) / sqrt(var + eps) * gamma + beta
def Cherry_LayerNormOp : Cherry_Op<"layernorm", [Pure]> {
  let summary = "Layer normalization";
  let arguments = (ins AnyCherryTensor:$input, AnyCherryTensor:$gamma, AnyCherryTensor:$beta, F32Attr:$eps);
  let results = (outs AnyCherryTensor:$result);
  let assemblyFormat = "$input `,` $gamma `,` $beta `eps` $eps attr-dict `:` functional-type(operands, results)";
}

// 4. Broadcast: 将 Tensor 广播到新形状
// 例如: [1] -> [128, 128]
def Cherry_BroadcastOp : Cherry_Op<"broadcast", [Pure]> {
  let summary = "Broadcast tensor to new shape";
  let arguments = (ins AnyCherryTensor:$input, Variadic<I64>:$target_shape);
  let results = (outs AnyCherryTensor:$result);
  let assemblyFormat = "$input `,` $target_shape attr-dict `:` `(` type($input) `,` type($target_shape) `)` `->` type($result)";
}

#endif
