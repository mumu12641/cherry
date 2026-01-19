module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %x0 : tensor<10x10xf32>
  }

  func.func @main() {
    %cst = arith.constant 2.0 : f32
    %input = tensor.splat %cst : tensor<10x10xf32>
    %init = tensor.empty() : tensor<10x10xf32>
    %result = func.call @square(%input, %init) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %buffer = bufferization.to_memref %result : memref<10x10xf32>
    %U = memref.cast %buffer : memref<10x10xf32> to memref<*xf32>
    func.call @printMemrefF32(%U) : (memref<*xf32>) -> ()
    return
  }
}
