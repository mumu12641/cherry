#include "Conversion/GPU/InsertHostRegister.h"

#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"


namespace mlir::cherry {
#define GEN_PASS_DEF_CHERRYINSERTHOSTREGISTERPASS
#include "Conversion/Passes.h.inc"
}   // namespace mlir::cherry
using namespace mlir;
using namespace mlir::cherry;
struct CherryInsertHostRegisterPass
    : mlir::cherry::impl::CherryInsertHostRegisterPassBase<CherryInsertHostRegisterPass>
{
    using mlir::cherry::impl::CherryInsertHostRegisterPassBase<
        CherryInsertHostRegisterPass>::CherryInsertHostRegisterPassBase;
    void runOnOperation() override;
};

void CherryInsertHostRegisterPass::runOnOperation()
{
    func::FuncOp func = getOperation();

    // 收集需要处理的 AllocOp
    SmallVector<memref::AllocOp, 4> allocs;
    func.walk([&](memref::AllocOp op) { allocs.push_back(op); });

    for (auto allocOp : allocs) {
        OpBuilder builder(allocOp);
        builder.setInsertionPointAfter(allocOp);

        Value allocResult = allocOp.getResult();
        // auto  rankedType  = allocResult.getType().cast<MemRefType>();
        auto rankedType = llvm::cast<MemRefType>(allocResult.getType());
        auto elemType   = rankedType.getElementType();

        // 创建 Unranked 类型: memref<*xf32>
        auto unrankedType = UnrankedMemRefType::get(elemType, rankedType.getMemorySpace());

        // 1. 插入 Cast
        auto castOp = builder.create<memref::CastOp>(allocOp.getLoc(), unrankedType, allocResult);

        // 2. 插入 HostRegister
        builder.create<gpu::HostRegisterOp>(allocOp.getLoc(), castOp.getResult());
    }
}
