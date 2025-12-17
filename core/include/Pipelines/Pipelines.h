#ifndef CHERRY_PIPELINES_H
#define CHERRY_PIPELINES_H
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::pipeline {
struct CherryPipelineOptions : public PassPipelineOptions<CherryPipelineOptions>
{
};

void buildCherryBasicPipeline(OpPassManager& pm, const CherryPipelineOptions& options);
void registerCherryBasicPipelines();
void registerCherryBasicPipelinesExtension(mlir::DialectRegistry& registry);

// Phase 1: Inlining
void addInlinerPass(mlir::OpPassManager& pm);

// Phase 2: Shape Inference
void addShapeInferencePass(mlir::OpPassManager& pm);

// Phase 3: Canonicalization
void addCanoicalizerPass(mlir::OpPassManager& pm);

// Phase 4: Linalg Conversion
void addLinalgConversionPass(mlir::OpPassManager& pm);

// Phase 5: Linalg Tiling & Fusion
void addLinalgTilingPass(mlir::OpPassManager& pm);

// Phase 6: Bufferization
void addBufferizationPass(mlir::OpPassManager& pm);

// Phase 7: Lowering to SCF/Loops
void addLinalgToSCFPass(mlir::OpPassManager& pm);

// Phase 8: Lowering to LLVM
void addLLVMLoweringPass(mlir::OpPassManager& pm);

void addLinalgVectorizationPass(mlir::OpPassManager& pm);
}   // namespace mlir::pipeline
#endif
