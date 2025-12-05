#include "Conversion/CherryToLinalg/CherryToLinalg.h"
#include "Conversion/CherryToLinalg/LinalgTiling.h"
#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "Dialect/Cherry/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// 辅助函数：运行 Pass 并在成功后打印到文件
// 这样可以避免写很多重复的 if(failed) 代码
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
    registry.insert<mlir::cherry::CherryDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::math::MathDialect>();
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

    if (!runPipelineAndPrint("test", *module, fileStream, [](mlir::PassManager& pm) {
        }))
        return 1;

    llvm::outs() << "Processing complete. Output saved to: " << outputFilename << "\n";
    return 0;
}
