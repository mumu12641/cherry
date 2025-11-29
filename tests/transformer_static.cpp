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

// 辅助函数：创建 Transformer 示例
void example_transformer(mlir::OpBuilder& builder, mlir::MLIRContext& context,
                         mlir::ModuleOp module)
{
    llvm::outs() << "\n=== Generating Transformer Block Example ===\n";

    auto loc     = builder.getUnknownLoc();
    auto f32Type = builder.getF32Type();
    auto i64Type = builder.getI64Type();

    // 1. 定义维度常量
    const int64_t B  = 1;    // Batch Size
    const int64_t S  = 4;    // Sequence Length
    const int64_t D  = 8;    // Model Dimension (d_model)
    const int64_t FF = 32;   // Feed Forward Dimension

    // 2. 定义 Tensor 类型
    // Input/Output: [B, S, D]
    auto typeInput = mlir::cherry::CherryTensorType::get(&context, {B, S, D}, f32Type);
    // Weights Q/K/V: [D, D]
    auto typeWeightAttn = mlir::cherry::CherryTensorType::get(&context, {D, D}, f32Type);
    // Weights FFN: [D, FF] 和 [FF, D]
    auto typeWeightFF1 = mlir::cherry::CherryTensorType::get(&context, {D, FF}, f32Type);
    auto typeWeightFF2 = mlir::cherry::CherryTensorType::get(&context, {FF, D}, f32Type);
    // LayerNorm Params: [D]
    auto typeLnParam = mlir::cherry::CherryTensorType::get(&context, {D}, f32Type);

    // 中间变量类型
    // Attention Scores: [B, S, S]
    auto typeScore = mlir::cherry::CherryTensorType::get(&context, {B, S, S}, f32Type);
    // Transposed K: [B, D, S] (假设 K 是 [B, S, D])
    // 注意：这里的 MatMul 逻辑取决于具体的实现。
    // 假设 MatMul 支持 [B, S, D] * [D, D] -> [B, S, D]
    // 计算 Score 时需要 [B, S, D] * [B, D, S] -> [B, S, S]
    auto typeKTrans = mlir::cherry::CherryTensorType::get(&context, {B, D, S}, f32Type);
    // FFN Hidden: [B, S, FF]
    auto typeFFHidden = mlir::cherry::CherryTensorType::get(&context, {B, S, FF}, f32Type);

    // 3. 创建函数
    auto funcType = builder.getFunctionType({typeInput,
                                             typeWeightAttn,
                                             typeWeightAttn,
                                             typeWeightAttn,
                                             typeWeightFF1,
                                             typeWeightFF2,
                                             typeLnParam,
                                             typeLnParam},
                                            {typeInput});
    auto funcOp   = builder.create<mlir::func::FuncOp>(loc, "simple_transformer_block", funcType);

    mlir::Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // 获取参数
    auto        args     = entryBlock->getArguments();
    mlir::Value x        = args[0];
    mlir::Value w_q      = args[1];
    mlir::Value w_k      = args[2];
    mlir::Value w_v      = args[3];
    mlir::Value w_ff1    = args[4];
    mlir::Value w_ff2    = args[5];
    mlir::Value ln_gamma = args[6];
    mlir::Value ln_beta  = args[7];

    // === 辅助 Lambda ===
    auto createI64 = [&](int64_t val) {
        return builder.create<mlir::cherry::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(val));
    };

    // 创建一个标量 Tensor 常量 (用于广播)
    auto createScalarTensorConst = [&](float val) {
        // 1. 准备数据: 使用 Builtin RankedTensorType 创建 Attribute
        // 这是一个标准的 MLIR 构造过程，用于存储数据
        auto builtinType = mlir::RankedTensorType::get({1}, f32Type);
        auto attr        = mlir::DenseElementsAttr::get(builtinType, llvm::ArrayRef<float>{val});

        // 2. 准备结果类型: CherryTensorType
        auto cherryType = mlir::cherry::CherryTensorType::get(&context, {1}, f32Type);

        // 3. 创建 Op: 使用新的 cherry.create_tensor
        // 注意：这里使用的是 CreateTensorOp
        return builder.create<mlir::cherry::CreateTensorOp>(loc,
                                                            cherryType,   // 结果类型
                                                            attr          // 数据属性
        );
    };

    auto createTensorConst =
        [&](llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<float> data) -> mlir::Value {
        auto builtinType = mlir::RankedTensorType::get(shape, f32Type);

        // 简单的校验
        int64_t numElements = 1;
        for (auto dim : shape) numElements *= dim;
        if (data.size() != numElements) {
            llvm::errs() << "Error: Data size " << data.size() << " does not match shape size "
                         << numElements << "\n";
            return nullptr;
        }

        auto attr       = mlir::DenseElementsAttr::get(builtinType, data);
        auto cherryType = mlir::cherry::CherryTensorType::get(&context, shape, f32Type);

        return builder.create<mlir::cherry::CreateTensorOp>(loc, cherryType, attr);
    };

    // ==========================================
    // 1. Self-Attention
    // ==========================================

    // 1.1 Q, K, V 投影
    auto q = builder.create<mlir::cherry::MatMulOp>(loc, typeInput, x, w_q);
    auto k = builder.create<mlir::cherry::MatMulOp>(loc, typeInput, x, w_k);
    auto v = builder.create<mlir::cherry::MatMulOp>(loc, typeInput, x, w_v);

    // 1.2 转置 K -> K^T
    // 假设输入是 [1, 4, 8]，我们要交换最后两维变成 [1, 8, 4]
    // Permutation: 0, 2, 1
    auto p0 = createI64(0);
    auto p1 = createI64(2);
    auto p2 = createI64(1);
    auto k_t =
        builder.create<mlir::cherry::TransposeOp>(loc, typeKTrans, k, mlir::ValueRange{p0, p1, p2});

    // 1.3 计算 Attention Score = Q * K^T
    auto scores = builder.create<mlir::cherry::MatMulOp>(loc, typeScore, q, k_t);

    // 1.4 缩放 (Divide by sqrt(D))
    // sqrt(8) approx 2.828
    auto sqrt_dk   = createScalarTensorConst(2.8284f);
    auto sqrt_dk_1 = createTensorConst({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // 广播到 [B, S, S]
    auto d0           = createI64(B);
    auto d1           = createI64(S);
    auto d2           = createI64(S);   // Score shape is [B, S, S]
    auto scale_tensor = builder.create<mlir::cherry::BroadcastOp>(
        loc, typeScore, sqrt_dk, mlir::ValueRange{d0, d1, d2});

    auto scores_scaled =
        builder.create<mlir::cherry::TensorDivOp>(loc, typeScore, scores, scale_tensor);

    // 1.5 Softmax (axis = 2)
    auto attn_weights = builder.create<mlir::cherry::SoftmaxOp>(
        loc, typeScore, scores_scaled, builder.getI64IntegerAttr(2));

    // 1.6 Output = Weights * V
    auto attn_out = builder.create<mlir::cherry::MatMulOp>(loc, typeInput, attn_weights, v);

    // ==========================================
    // 2. Add & Norm 1
    // ==========================================
    auto res1 = builder.create<mlir::cherry::TensorAddOp>(loc, typeInput, x, attn_out);

    // LayerNorm (eps = 1e-5)
    auto norm1 = builder.create<mlir::cherry::LayerNormOp>(
        loc, typeInput, res1, ln_gamma, ln_beta, builder.getF32FloatAttr(1e-5f));

    // ==========================================
    // 3. Feed Forward Network (FFN)
    // ==========================================

    // Linear 1: [B, S, D] * [D, FF] -> [B, S, FF]
    auto ff1 = builder.create<mlir::cherry::MatMulOp>(loc, typeFFHidden, norm1, w_ff1);

    // ReLU
    auto ff1_relu = builder.create<mlir::cherry::TensorReluOp>(loc, typeFFHidden, ff1);

    // Linear 2: [B, S, FF] * [FF, D] -> [B, S, D]
    auto ff2 = builder.create<mlir::cherry::MatMulOp>(loc, typeInput, ff1_relu, w_ff2);

    // ==========================================
    // 4. Add & Norm 2
    // ==========================================
    auto res2 = builder.create<mlir::cherry::TensorAddOp>(loc, typeInput, norm1, ff2);

    auto final_output = builder.create<mlir::cherry::LayerNormOp>(
        loc, typeInput, res2, ln_gamma, ln_beta, builder.getF32FloatAttr(1e-5f));

    // Return
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{final_output});
}

int main()
{
    // 1. 初始化
    mlir::DialectRegistry registry;
    mlir::MLIRContext     context(registry);
    context.loadAllAvailableDialects();

    auto dialect = context.getOrLoadDialect<mlir::cherry::CherryDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    dialect->registerType();
    dialect->registerOps();

    mlir::OpBuilder                   builder(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    // 2. 调用 Transformer 示例生成函数
    example_transformer(builder, context, *module);

    // 3. 打印结果
    module->print(llvm::outs());
    std::string     filename = "/home/nx/ycy/pb/cherry/core/transformer.mlir";
    std::error_code ec;

    // 创建文件流
    // 参数: 文件名, 错误码, 打开模式(默认即可)
    llvm::raw_fd_ostream fileStream(filename, ec);

    if (ec) {
        llvm::errs() << "无法打开文件: " << ec.message() << "\n";
        return 1;
    }

    // 将 Module 打印到文件流中
    module->print(fileStream);

    llvm::outs() << "MLIR 代码已成功保存到: " << filename << "\n";

    return 0;

    return 0;
}