
#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Pipelines/Pipelines.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <system_error>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"), cl::init("output.ll"));

int main(int argc, char** argv)
{
    cl::ParseCommandLineOptions(argc, argv, "Cherry Compiler Driver\n");

    mlir::DialectRegistry registry;
    registry.insert<mlir::cherry::CherryDialect,
                    mlir::arith::ArithDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::scf::SCFDialect,
                    mlir::func::FuncDialect,
                    mlir::tensor::TensorDialect,
                    mlir::memref::MemRefDialect,
                    mlir::vector::VectorDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::affine::AffineDialect>();

    mlir::pipeline::registerCherryBasicPipelinesExtension(registry);

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    // ============================================================
    // Parsing input file
    // ============================================================
    llvm::outs() << "🚀 Parsing input file: " << inputFilename << " ...\n";

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);

    if (!module) {
        llvm::errs() << "❌ Error: Failed to parse input file: " << inputFilename << "\n";
        return 1;
    }
    llvm::outs() << "✅ Successfully parsed input.\n";

    // ============================================================
    // Running Cherry Optimization Pipeline
    // ============================================================
    llvm::outs() << "⚙️ Running Cherry Optimization Pipeline...\n";

    mlir::PassManager pm(&context);
    pm.enableVerifier();

    mlir::pipeline::CherryPipelineOptions options;
    mlir::pipeline::buildCherryBasicPipeline(pm, options);

    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "💥 Pipeline execution failed!\n";
        return 1;
    }
    llvm::outs() << "✨ Pipeline finished successfully.\n";

    // ============================================================
    // Translating to LLVM IR
    // ============================================================
    llvm::outs() << "🔨 Translating to LLVM IR...\n";

    llvm::LLVMContext             llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

    if (!llvmModule) {
        llvm::errs() << "❌ Error: Failed to translate module to LLVM IR.\n";
        return 1;
    }

    llvmModule->setDataLayout(
        "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
    llvmModule->setTargetTriple("x86_64-unknown-linux-gnu");

    std::error_code      ec;
    llvm::raw_fd_ostream fileStream(outputFilename, ec);
    if (ec) {
        llvm::errs() << "❌ Error: Cannot open output file '" << outputFilename
                     << "': " << ec.message() << "\n";
        return 1;
    }

    llvmModule->print(fileStream, nullptr);
    llvm::outs() << "📦 LLVM IR saved to: " << outputFilename << "\n";

    return 0;
}
