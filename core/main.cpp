#include "Conversion/CherryToLinalg/CherryToLinalg.h"
#include "Conversion/CherryToLinalg/LinalgTiling.h"
#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "Dialect/Cherry/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <system_error>
#include <utility>

bool runPipelineAndPrint(const std::string& phaseName, mlir::ModuleOp module,
                         llvm::raw_fd_ostream&                   fileStream,
                         std::function<void(mlir::PassManager&)> addPassesFunc)
{

    mlir::PassManager pm(module.getContext());
    pm.enableVerifier();

    addPassesFunc(pm);

    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Failed to run pass pipeline: " << phaseName << "\n";
        return false;
    }

    fileStream << "\n// ==========================================\n";
    fileStream << "// Phase: " << phaseName << "\n";
    fileStream << "// ==========================================\n";
    module.print(fileStream);
    fileStream.flush();

    return true;
}

int main(int argc, char** argv)
{
    mlir::DialectRegistry registry;
    registry.insert<mlir::cherry::CherryDialect,
                    mlir::arith::ArithDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::scf::SCFDialect,
                    mlir::func::FuncDialect,
                    mlir::tensor::TensorDialect,
                    mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::affine::AffineDialect>();

    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
    // mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::func::registerAllExtensions(registry);

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    std::string inputFilename  = "/home/nx/ycy/pb/cherry/tests/test_transformer.mlir";
    std::string outputFilename = "/home/nx/ycy/pb/cherry/tests/test_transformer_output.mlir";

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);

    if (!module) {
        llvm::errs() << "Error: Failed to parse input file: " << inputFilename << "\n";
        return 1;
    }
    llvm::outs() << "Successfully parsed: " << inputFilename << "\n";

    std::error_code      ec;
    llvm::raw_fd_ostream fileStream(outputFilename, ec);
    if (ec) {
        llvm::errs() << "Error: Cannot open output file: " << ec.message() << "\n";
        return 1;
    }

    fileStream << "// Original IR loaded from file\n";
    module->print(fileStream);


    // Phase 1: Inlining
    if (!runPipelineAndPrint("Inliner", *module, fileStream, [](mlir::PassManager& pm) {
            pm.addPass(mlir::createInlinerPass());
        }))
        return 1;

    // Phase 2: Type/Shape Inference
    if (!runPipelineAndPrint("Shape Inference", *module, fileStream, [](mlir::PassManager& pm) {
            mlir::OpPassManager& optPM = pm.nest<mlir::cherry::FuncOp>();
            optPM.addPass(mlir::cherry::createCherryShapeInferencePass());
        }))
        return 1;

    // Phase 3: Canonicalizer
    if (!runPipelineAndPrint("Canonicalizer", *module, fileStream, [](mlir::PassManager& pm) {
            pm.addPass(mlir::createCanonicalizerPass());
        }))
        return 1;


    // Phase 4: Convert to Linalg
    if (!runPipelineAndPrint("Convert to Linalg", *module, fileStream, [](mlir::PassManager& pm) {
            pm.addPass(mlir::cherry::createConvertCherryToLinalgPass());
        }))
        return 1;

    // Phase 5: Linalg Tiling
    if (!runPipelineAndPrint("Linalg Tiling", *module, fileStream, [](mlir::PassManager& pm) {
            mlir::OpPassManager& optPM = pm.nest<mlir::func::FuncOp>();
            optPM.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
            optPM.addPass(mlir::createLinalgElementwiseOpFusionPass());
            optPM.addPass(mlir::cherry::createCherryLinalgTilingPass());
        }))
        return 1;

    if (!runPipelineAndPrint("Bufferization", *module, fileStream, [](mlir::PassManager& pm) {
            pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

            mlir::bufferization::OneShotBufferizationOptions options;
            options.bufferizeFunctionBoundaries = true;
            pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
        }))
        return 1;

    if (!runPipelineAndPrint("linalg to scf", *module, fileStream, [](mlir::PassManager& pm) {
            // pm.addPass(mlir::createLinalgVectorizationPass());
            pm.addPass(mlir::createConvertLinalgToLoopsPass());
        }))
        return 1;
    if (!runPipelineAndPrint("lower to llvm", *module, fileStream, [](mlir::PassManager& pm) {
            // 1. SCF -> Control Flow (Branch/Jump)
            pm.addPass(mlir::createConvertSCFToCFPass());
            // 2. MemRef -> LLVM 
            pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
            // 3. Math/Arith -> LLVM
            pm.addPass(mlir::createConvertMathToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            // 4. Func -> LLVM
            pm.addPass(mlir::createConvertFuncToLLVMPass());
            // 5. remove Casts
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        }))
        return 1;




    llvm::outs() << "Processing complete. Output saved to: " << outputFilename << "\n";
    return 0;
}
