#include "dialect/cherry/IR/CherryDialect.h"
#include "dialect/cherry/IR/CherryOps.h"
#include "dialect/cherry/IR/CherryTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

mlir::Value createTensorConst(mlir::OpBuilder& builder, mlir::MLIRContext* context,
                              llvm::ArrayRef<int64_t> shape, float initValue)
{
    auto    loc         = builder.getUnknownLoc();
    auto    f32Type     = builder.getF32Type();
    int64_t numElements = 1;
    for (auto dim : shape) numElements *= dim;
    std::vector<float> data(numElements, initValue);
    auto               builtinType = mlir::RankedTensorType::get(shape, f32Type);
    auto attr       = mlir::DenseElementsAttr::get(builtinType, llvm::ArrayRef<float>(data));
    auto cherryType = mlir::cherry::CherryTensorType::get(context, shape, f32Type);
    return builder.create<mlir::cherry::CreateTensorOp>(loc, cherryType, attr);
}

mlir::Value createI64Const(mlir::OpBuilder& builder, int64_t val)
{
    return builder.create<mlir::cherry::ConstantOp>(
        builder.getUnknownLoc(), builder.getI64Type(), builder.getI64IntegerAttr(val));
}

// ============================================================
// 1. 构建 Transformer Block 函数 (定义计算图，参数为动态形状)
// ============================================================
mlir::func::FuncOp buildTransformerBlock(mlir::OpBuilder& builder, mlir::MLIRContext& context)
{
    auto loc     = builder.getUnknownLoc();
    auto f32Type = builder.getF32Type();

    // 定义常量：Hidden Size 固定，但 Batch 和 Seq 动态
    const int64_t D   = 8;
    const int64_t FF  = 32;
    const int64_t Dyn = mlir::ShapedType::kDynamic;   // 代表 '?'

    // --- 定义参数类型 (Dynamic Shapes) ---
    // Input: [?, ?, 8]
    auto typeInput = mlir::cherry::CherryTensorType::get(&context, {Dyn, Dyn, D}, f32Type);
    // Weights: [8, 8] (固定)
    auto typeWeightAttn = mlir::cherry::CherryTensorType::get(&context, {D, D}, f32Type);
    // FFN Weights
    auto typeWeightFF1 = mlir::cherry::CherryTensorType::get(&context, {D, FF}, f32Type);
    auto typeWeightFF2 = mlir::cherry::CherryTensorType::get(&context, {FF, D}, f32Type);
    // LayerNorm Params
    auto typeLnParam = mlir::cherry::CherryTensorType::get(&context, {D}, f32Type);

    // --- 创建函数 ---
    auto funcType = builder.getFunctionType({typeInput,
                                             typeWeightAttn,
                                             typeWeightAttn,
                                             typeWeightAttn,
                                             typeWeightFF1,
                                             typeWeightFF2,
                                             typeLnParam,
                                             typeLnParam},
                                            {typeInput}   // Return type
    );

    auto funcOp = builder.create<mlir::func::FuncOp>(loc, "simple_transformer_block", funcType);
    mlir::Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // 获取参数
    auto        args = entryBlock->getArguments();
    mlir::Value x    = args[0];
    mlir::Value w_q = args[1], w_k = args[2], w_v = args[3];
    mlir::Value w_ff1 = args[4], w_ff2 = args[5];
    mlir::Value ln_gamma = args[6], ln_beta = args[7];

    // --- 内部类型推导 (为了演示，这里我们手动指定结果类型为 Dynamic) ---
    // 注意：在实际编译器中，这里通常需要 ShapeInferenceInterface
    auto typeDyn3D    = mlir::cherry::CherryTensorType::get(&context, {Dyn, Dyn, D}, f32Type);
    auto typeDyn3D_FF = mlir::cherry::CherryTensorType::get(&context, {Dyn, Dyn, FF}, f32Type);
    auto typeDynScore =
        mlir::cherry::CherryTensorType::get(&context, {Dyn, Dyn, Dyn}, f32Type);   // [B, S, S]

    // 1. Self-Attention
    auto q = builder.create<mlir::cherry::MatMulOp>(loc, typeDyn3D, x, w_q);
    auto k = builder.create<mlir::cherry::MatMulOp>(loc, typeDyn3D, x, w_k);
    auto v = builder.create<mlir::cherry::MatMulOp>(loc, typeDyn3D, x, w_v);

    // Transpose K
    auto p0 = createI64Const(builder, 0);
    auto p1 = createI64Const(builder, 2);
    auto p2 = createI64Const(builder, 1);
    // K^T shape: [?, 8, ?]
    auto typeKTrans = mlir::cherry::CherryTensorType::get(&context, {Dyn, D, Dyn}, f32Type);
    auto k_t =
        builder.create<mlir::cherry::TransposeOp>(loc, typeKTrans, k, mlir::ValueRange{p0, p1, p2});

    // Score
    auto scores = builder.create<mlir::cherry::MatMulOp>(loc, typeDynScore, q, k_t);

    // Scale & Softmax
    // 注意：这里为了简单，我们假设 Broadcast
    // 的目标形状由外部逻辑保证，或者我们在这里硬编码为运行时预期的形状
    // 在完美的动态实现中，这里应该用 tensor.dim 获取 x 的维度。
    // 这里为了代码跑通，我们先创建标量 sqrt_dk
    auto sqrt_dk = createTensorConst(builder, &context, {1}, 2.8284f);

    // 这是一个 Hack：为了演示，我们广播到一个非常大的动态范围，或者假设后端能处理
    // 更好的做法是引入 DimOp，但为了不引入更多 Op，我们这里假设 Broadcast 接受动态 shape 定义
    // 或者我们简单地只做除法（假设 tensor_div 支持标量广播，如果不支持，这里代码会比较复杂）
    // 让我们假设 tensor_div 支持隐式广播，直接除以 sqrt_dk
    // 如果不支持，我们需要 BroadcastOp，这里暂时用占位符
    auto d0 = createI64Const(
        builder, 1);   // Batch (Runtime should handle mismatch if broadcast supports it)
    auto d1           = createI64Const(builder, 4);
    auto scale_tensor = builder.create<mlir::cherry::BroadcastOp>(
        loc, typeDynScore, sqrt_dk, mlir::ValueRange{d0, d1, d1});

    auto scores_scaled =
        builder.create<mlir::cherry::TensorDivOp>(loc, typeDynScore, scores, scale_tensor);
    auto attn_weights = builder.create<mlir::cherry::SoftmaxOp>(
        loc, typeDynScore, scores_scaled, builder.getI64IntegerAttr(2));
    auto attn_out = builder.create<mlir::cherry::MatMulOp>(loc, typeDyn3D, attn_weights, v);

    // 2. Add & Norm
    auto res1  = builder.create<mlir::cherry::TensorAddOp>(loc, typeDyn3D, x, attn_out);
    auto norm1 = builder.create<mlir::cherry::LayerNormOp>(
        loc, typeDyn3D, res1, ln_gamma, ln_beta, builder.getF32FloatAttr(1e-5));

    // 3. FFN
    auto ff1      = builder.create<mlir::cherry::MatMulOp>(loc, typeDyn3D_FF, norm1, w_ff1);
    auto ff1_relu = builder.create<mlir::cherry::TensorReluOp>(loc, typeDyn3D_FF, ff1);
    auto ff2      = builder.create<mlir::cherry::MatMulOp>(loc, typeDyn3D, ff1_relu, w_ff2);

    // 4. Add & Norm
    auto res2         = builder.create<mlir::cherry::TensorAddOp>(loc, typeDyn3D, norm1, ff2);
    auto final_output = builder.create<mlir::cherry::LayerNormOp>(
        loc, typeDyn3D, res2, ln_gamma, ln_beta, builder.getF32FloatAttr(1e-5));

    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{final_output});

    return funcOp;
}

// ============================================================
// 2. 构建 Main 函数 (定义具体数据，调用 Block)
// ============================================================
void buildMain(mlir::OpBuilder& builder, mlir::MLIRContext& context,
                                             mlir::func::FuncOp callee)
{
    auto loc = builder.getUnknownLoc();

    // 定义具体的 Static Shapes
    const int64_t B  = 1;
    const int64_t S  = 4;
    const int64_t D  = 8;
    const int64_t FF = 32;

    auto funcType = builder.getFunctionType({}, {});   // main() -> void
    auto funcOp   = builder.create<mlir::func::FuncOp>(loc, "main", funcType);

    mlir::Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    llvm::outs() << "Creating Concrete Tensors in Main...\n";

    // 1. 创建具体数据 (Inputs)
    auto input = createTensorConst(builder, &context, {B, S, D}, 0.5f);   // Input X

    // 2. 创建具体权重 (Weights)
    auto w_q = createTensorConst(builder, &context, {D, D}, 0.1f);
    auto w_k = createTensorConst(builder, &context, {D, D}, 0.1f);
    auto w_v = createTensorConst(builder, &context, {D, D}, 0.1f);

    auto w_ff1 = createTensorConst(builder, &context, {D, FF}, 0.2f);
    auto w_ff2 = createTensorConst(builder, &context, {FF, D}, 0.2f);

    auto gamma = createTensorConst(builder, &context, {D}, 1.0f);
    auto beta  = createTensorConst(builder, &context, {D}, 0.0f);

    // 3. 调用 Transformer Block
    // CallOp 需要 Callee 的名字和输入参数
    mlir::ValueRange args   = {input, w_q, w_k, w_v, w_ff1, w_ff2, gamma, beta};
    auto             callOp = builder.create<mlir::func::CallOp>(loc, callee, args);

    // 4. 获取结果 (可选：打印结果或返回)
    // 这里我们只是演示调用，所以直接 Return
    builder.create<mlir::func::ReturnOp>(loc);
}

int main()
{
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::cherry::CherryDialect>(); 
    mlir::MLIRContext     context(registry);
    context.loadAllAvailableDialects();

    auto dialect = context.getOrLoadDialect<mlir::cherry::CherryDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    dialect->registerType();
    dialect->registerOps();

    mlir::OpBuilder                   builder(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    auto transformerFunc = buildTransformerBlock(builder, context);
    builder.setInsertionPointToEnd(module->getBody());
    buildMain(builder, context, transformerFunc);

    module->print(llvm::outs());

    std::string          filename = "/home/nx/ycy/pb/cherry/core/test.mlir";
    std::error_code      ec;
    llvm::raw_fd_ostream fileStream(filename, ec);
    if (ec) {
        llvm::errs() << "无法打开文件: " << ec.message() << "\n";
        return 1;
    }
    module->print(fileStream);
    llvm::outs() << "MLIR 代码已成功保存到: " << filename << "\n";
    return 0;
}