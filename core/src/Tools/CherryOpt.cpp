#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/Transforms/Passes.h"
#include "Conversion/Passes.h"
#include "Pipelines/Pipelines.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv)
{
    mlir::registerAllPasses();

    mlir::pipeline::registerCherryBasicPipelines();

    mlir::DialectRegistry registry;

    mlir::registerAllDialects(registry);

    registry.insert<mlir::cherry::CherryDialect>();

    mlir::pipeline::registerCherryBasicPipelinesExtension(registry);
    mlir::cherry::registerCherryOptPasses();
    mlir::cherry::registerCherryConversionPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Cherry Optimizer Driver", registry));
}
