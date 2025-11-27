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

#include "llvm/Support/raw_ostream.h"

#include <iostream>

void example()
{
    mlir::DialectRegistry registry;
    mlir::MLIRContext     context(registry);
    context.loadAllAvailableDialects();

    auto dialect = context.getOrLoadDialect<mlir::cherry::CherryDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    // 注册类型和操作
    dialect->registerType();
    dialect->registerOps();

    mlir::OpBuilder                   builder(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    // 2. 准备类型
    auto f32Type = builder.getF32Type();
    auto i64Type = builder.getI64Type();

    // 输入 Tensor: [2, 3]
    auto inputType = mlir::cherry::CherryTensorType::get(&context, {2, 3}, f32Type);

    // 转置/Reshape 后的 Tensor: [3, 2]
    auto transType = mlir::cherry::CherryTensorType::get(&context, {3, 2}, f32Type);

    // 归约后的 Tensor (沿轴 1 归约: [2, 3] -> [2])
    auto reducedTensorType = mlir::cherry::CherryTensorType::get(&context, {2}, f32Type);

    llvm::outs() << "Creating Operations...\n";

    // 3. 定义函数 test_all_ops
    // 为了简单起见，我们让函数返回 void，只在函数体内生成 Op 用于演示
    auto funcType = builder.getFunctionType({inputType, inputType}, {});
    auto funcOp =
        builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "test_all_ops", funcType);

    mlir::Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // 获取参数
    mlir::Value lhs = entryBlock->getArgument(0);
    mlir::Value rhs = entryBlock->getArgument(1);
    auto        loc = builder.getUnknownLoc();

    // ==========================================
    // 1. Binary Ops (二元操作)
    // ==========================================
    auto add = builder.create<mlir::cherry::TensorAddOp>(loc, inputType, lhs, rhs);
    auto sub = builder.create<mlir::cherry::TensorSubOp>(loc, inputType, lhs, rhs);
    auto mul = builder.create<mlir::cherry::TensorMulOp>(loc, inputType, lhs, rhs);
    // 注意：请确保你在 td 文件里修正了 Div 的名字
    auto div = builder.create<mlir::cherry::TensorDivOp>(loc, inputType, lhs, rhs);

    // ==========================================
    // 2. Unary Ops (一元操作)
    // ==========================================
    auto neg  = builder.create<mlir::cherry::TensorNegOp>(loc, inputType, lhs);
    auto exp  = builder.create<mlir::cherry::TensorExpOp>(loc, inputType, lhs);
    auto relu = builder.create<mlir::cherry::TensorReluOp>(loc, inputType, lhs);
    auto sig  = builder.create<mlir::cherry::TensorSigmoidOp>(loc, inputType, lhs);
    auto tanh = builder.create<mlir::cherry::TensorTanhOp>(loc, inputType, lhs);

    // ==========================================
    // 3. Reduction Ops (归约操作)
    // ==========================================

    // 辅助函数：创建一个 i64 常量 (用于 axis, shape 等参数)
    auto createI64Const = [&](int64_t val) -> mlir::Value {
        return builder.create<mlir::cherry::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(val));
    };

    // 3.1 全局归约 (Full Reduction) -> 结果是标量 f32
    // Optional<I64>:$axis 为空时，传入 mlir::Value()
    auto meanFull = builder.create<mlir::cherry::MeanOp>(loc, f32Type, lhs, mlir::Value());
    auto maxFull  = builder.create<mlir::cherry::MaxReduceOp>(loc, f32Type, lhs, mlir::Value());

    // 3.2 轴归约 (Axis Reduction) -> 结果是 Tensor
    // 沿轴 1 归约
    mlir::Value axis1   = createI64Const(1);
    auto        sumAxis = builder.create<mlir::cherry::SumOp>(loc, reducedTensorType, lhs, axis1);
    auto minAxis = builder.create<mlir::cherry::MinReduceOp>(loc, reducedTensorType, lhs, axis1);

    // Argmax 返回的是索引，通常是 i64 或 i32，这里假设是 i64
    // 结果类型应该是 Tensor<2xi64>
    auto indexTensorType = mlir::cherry::CherryTensorType::get(&context, {2}, i64Type);
    auto argmax          = builder.create<mlir::cherry::ArgmaxOp>(loc, indexTensorType, lhs, axis1);

    // ==========================================
    // 4. Shape/Memory Ops (形状操作)
    // ==========================================

    // 4.1 Reshape: [2, 3] -> [3, 2]
    // 参数是 Variadic<I64>，需要传入 ValueRange
    mlir::Value dim0 = createI64Const(3);
    mlir::Value dim1 = createI64Const(2);
    auto        reshape =
        builder.create<mlir::cherry::ReshapeOp>(loc, transType, lhs, mlir::ValueRange{dim0, dim1});

    // 4.2 Transpose: [2, 3] -> [3, 2] (交换维度 0 和 1)
    mlir::Value perm0     = createI64Const(1);   // 原来的第1维放到第0位
    mlir::Value perm1     = createI64Const(0);   // 原来的第0维放到第1位
    auto        transpose = builder.create<mlir::cherry::TransposeOp>(
        loc, transType, lhs, mlir::ValueRange{perm0, perm1});

    // 结束函数
    builder.create<mlir::func::ReturnOp>(loc);

    // 打印 IR
    module->print(llvm::outs());
    return;
}

void example_scalar()
{
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

    // 准备类型
    auto f32Type    = builder.getF32Type();
    auto i32Type    = builder.getI32Type();   // Integer 类型
    auto tensorType = mlir::cherry::CherryTensorType::get(&context, {2, 3}, f32Type);

    llvm::outs() << "Creating Operations...\n";

    // 定义函数
    auto funcType = builder.getFunctionType({tensorType}, {});
    auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "test_ops", funcType);

    mlir::Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value tensorInput = entryBlock->getArgument(0);
    auto        loc         = builder.getUnknownLoc();

    // ==========================================
    // 1. Scalar Operations (标量运算测试)
    // ==========================================

    // 1.1 创建标量常量 (使用 cherry.constant)
    // Float (f32)
    auto f32Val1 =
        builder.create<mlir::cherry::ConstantOp>(loc, f32Type, builder.getF32FloatAttr(10.5f));
    auto f32Val2 =
        builder.create<mlir::cherry::ConstantOp>(loc, f32Type, builder.getF32FloatAttr(2.5f));

    // Integer (i32)
    auto i32Val1 =
        builder.create<mlir::cherry::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(100));
    auto i32Val2 =
        builder.create<mlir::cherry::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(50));

    // 1.2 测试 Float 标量运算
    // 10.5 + 2.5
    auto sAdd = builder.create<mlir::cherry::ScalarAddOp>(loc, f32Type, f32Val1, f32Val2);
    // 10.5 - 2.5
    auto sSub = builder.create<mlir::cherry::ScalarSubOp>(loc, f32Type, f32Val1, f32Val2);

    // 1.3 测试 Integer 标量运算
    // 100 * 50
    auto sMul = builder.create<mlir::cherry::ScalarMulOp>(loc, i32Type, i32Val1, i32Val2);
    // 100 / 50
    auto sDiv = builder.create<mlir::cherry::ScalarDivOp>(loc, i32Type, i32Val1, i32Val2);


    // ==========================================
    // 2. Tensor Operations (保留之前的测试)
    // ==========================================
    // 简单的 Tensor Add 演示
    auto tAdd =
        builder.create<mlir::cherry::TensorAddOp>(loc, tensorType, tensorInput, tensorInput);

    builder.create<mlir::func::ReturnOp>(loc);

    module->print(llvm::outs());

    return;
}

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

    return 0;
}
// int main()
// {
//     std::cout<<"********start example********"<<std::endl;
//     example();
//     std::cout<<"*********end example*********"<<std::endl;

//     std::cout<<"********start example_scalar********"<<std::endl;
//     example_scalar();
//     std::cout<<"*********end example_scalar*********"<<std::endl;

//     std::cout<<"********start example_transformer********"<<std::endl;
//     example_transformer();
//     std::cout<<"*********end example_transformer*********"<<std::endl;
//     return 0;
// }