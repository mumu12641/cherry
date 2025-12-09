#include "Conversion/CherryToLinalg/CherryToLinalg.h"
#include "Conversion/CherryToLinalg/LinalgTiling.h"
#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "Dialect/Cherry/Transforms/Passes.h"
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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <system_error>
#include <utility>

// [新增] 定义命令行参数
namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

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
    // [新增] 解析命令行参数
    cl::ParseCommandLineOptions(argc, argv, "Cherry Compiler Driver\n");

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
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerBuiltinDialectTranslation(registry);


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

    llvm::SmallString<128> outputBuffer(inputFilename);
    llvm::sys::path::replace_extension(outputBuffer, "");
    std::string outputFilename = outputBuffer.str().str() + "_output.mlir";

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);

    if (!module) {
        llvm::errs() << "Error: Failed to parse input file: " << inputFilename << "\n";
        return 1;
    }
    llvm::outs() << "Successfully parsed: " << inputFilename << "\n";
    llvm::outs() << "Output will be saved to: " << outputFilename << "\n";

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
            options.allowReturnAllocsFromLoops  = true;
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
            // memref.subview / memref -> base pointer, aligned pointer, offset, sizes, strides
            pm.addPass(mlir::memref::createExpandStridedMetadataPass());

            // lower affine
            pm.addPass(mlir::createLowerAffinePass());

            // SCF ->  Basic Blocks & Branch
            pm.addPass(mlir::createConvertSCFToCFPass());

            pm.addPass(mlir::createConvertFuncToLLVMPass());
            pm.addPass(mlir::createConvertIndexToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
            pm.addPass(mlir::createConvertMathToLLVMPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        }))
        return 1;

    llvm::outs() << "Processing complete. Output saved to: " << outputFilename << "\n";

    llvm::outs() << "Translating to LLVM IR...\n";

    llvm::LLVMContext             llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    llvmModule->setDataLayout(
        "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
    llvmModule->setTargetTriple("x86_64-unknown-linux-gnu");
    llvm::SmallString<128> llvmOutputBuffer(outputFilename);
    llvm::sys::path::replace_extension(llvmOutputBuffer, "ll");
    std::string llvmOutputFilename = llvmOutputBuffer.str().str();

    std::error_code      ec_llvm;
    llvm::raw_fd_ostream llvmFileStream(llvmOutputFilename, ec_llvm);

    llvmModule->print(llvmFileStream, nullptr);
    llvm::outs() << "LLVM IR saved to: " << llvmOutputFilename << "\n";

    return 0;
}
