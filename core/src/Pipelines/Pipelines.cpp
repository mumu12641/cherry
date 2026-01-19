#include "Pipelines/Pipelines.h"

#include "Conversion/CherryToLinalg/CherryToLinalg.h"
#include "Conversion/CherryToLinalg/LinalgTiling.h"
#include "Conversion/CherryToLinalg/LinalgVectorization.h"
#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "Dialect/Cherry/Transforms/Passes.h"
#include "Utils/Cache.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"


namespace mlir::pipeline {

void addInlinerPass(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createInlinerPass());
}

void addShapeInferencePass(mlir::OpPassManager& pm)
{
    auto& funcPm = pm.nest<mlir::cherry::FuncOp>();
    funcPm.addPass(mlir::cherry::createCherryShapeInferencePass());
}

void addCanoicalizerPass(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
}

void addLinalgConversionPass(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::cherry::createConvertCherryToLinalgPass());
}

void addLinalgTilingPass(mlir::OpPassManager& pm)
{
    auto&             funcPm = pm.nest<mlir::func::FuncOp>();
    cherry::CacheInfo info;
    auto [l1_tile, l2_tile] = info.computeMatmulTileSizes(4);
    // funcPm.addPass(mlir::createLinalgElementwiseOpFusionPass());
    funcPm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
    // funcPm.addPass(mlir::createLinalgElementwiseOpFusionPass());
    funcPm.addPass(mlir::cherry::createCherryLinalgTilingPass(
        cherry::CherryLinalgTilingPassOptions{l1_tile, l2_tile, true, false}));

}

void addLinalgVectorizationPass(mlir::OpPassManager& pm)
{
    auto& funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::cherry::createCherryLinalgVectorizationPass());
    funcPm.addPass(mlir::createCanonicalizerPass());
    funcPm.addPass(mlir::createCSEPass());
}

void addBufferizationPass(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

    mlir::bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.allowReturnAllocsFromLoops  = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
}

void addLinalgToSCFPass(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLoopInvariantCodeMotionPass());
}

void addLLVMLoweringPass(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    // pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
    // pm.addPass(mlir::createConvertVectorToLLVMPass());

    pm.addPass(mlir::createConvertSCFToCFPass());

    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void buildCherryBasicPipeline(OpPassManager& pm, const CherryPipelineOptions& options)
{
    addInlinerPass(pm);
    addShapeInferencePass(pm);
    addCanoicalizerPass(pm);
    addLinalgConversionPass(pm);
    addLinalgTilingPass(pm);
    // addLinalgVectorizationPass(pm);
    addBufferizationPass(pm);
    addLinalgToSCFPass(pm);
    addLLVMLoweringPass(pm);
}

void registerCherryBasicPipelines()
{
    mlir::PassPipelineRegistration<CherryPipelineOptions>(
        "cherry-lowering-pipeline", "Run full lowering to LLVM", buildCherryBasicPipeline);
}
void registerCherryBasicPipelinesExtension(mlir::DialectRegistry& registry)
{
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::vector::registerSubsetOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::func::registerAllExtensions(registry);
}

}   // namespace mlir::pipeline
