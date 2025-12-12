#include "Conversion/CherryToLinalg/CherryToLinalg.h"

#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <vector>

namespace mlir::cherry {
#define GEN_PASS_DEF_CONVERTCHERRYTOLINALGPASS
#include "Conversion/Passes.h.inc"
}   // namespace mlir::cherry

using namespace mlir;
using namespace mlir::cherry;
class CherryTypeConverter : public TypeConverter
{
public:
    CherryTypeConverter()
    {
        addConversion([](Type type) { return type; });
        addConversion([](CherryTensorType type) -> Type {
            return RankedTensorType::get(type.getShape(), type.getElementType());
        });
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::ConstantOp lowering
// ===----------------------------------------------------------------------===//
struct ConstantOpLowering : public OpConversionPattern<ConstantOp>
{
    using OpConversionPattern<ConstantOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                       llvm::cast<mlir::TypedAttr>(op.getValue()));
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::CreateTensorOp lowering
// ===----------------------------------------------------------------------===//
struct CreateTensorOpLowering : public OpConversionPattern<CreateTensorOp>
{
    using OpConversionPattern<CreateTensorOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CreateTensorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto valueAttr  = op.getValue();
        auto targetType = typeConverter->convertType(op.getResult().getType());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, targetType, valueAttr);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ::mlir::cherry::WeightOpLowering lowering
//===----------------------------------------------------------------------===//
LogicalResult readTensorFromFile(Location loc, StringRef filepath, ArrayRef<int64_t> expectedShape,
                                 std::vector<char>& outRawBytes)
{
    std::ifstream f(filepath.str(), std::ios::binary);
    if (!f.is_open()) return emitError(loc, "Error: Cannot open weight file: ") << filepath;


    int ndim;
    if (!f.read(reinterpret_cast<char*>(&ndim), sizeof(int)))
        return emitError(loc, "Failed to read ndim");
    std::vector<int> fileShape(ndim);
    if (!f.read(reinterpret_cast<char*>(fileShape.data()), ndim * sizeof(int)))
        return emitError(loc, "Failed to read shape data");

    if (ndim != expectedShape.size())
        return emitError(loc) << "Shape rank mismatch. File: " << ndim
                              << ", Expected: " << expectedShape.size();

    int64_t numElements = 1;
    for (int i = 0; i < ndim; ++i) {
        if (fileShape[i] != expectedShape[i])
            return emitError(loc) << "Dimension mismatch at index " << i
                                  << ". File: " << fileShape[i]
                                  << ", Expected: " << expectedShape[i];

        numElements *= fileShape[i];
    }

    // Read Data
    size_t bytesToRead = numElements * sizeof(float);
    outRawBytes.resize(bytesToRead);

    if (!f.read(outRawBytes.data(), bytesToRead))
        return emitError(loc, "Failed to read tensor data (unexpected EOF)");

    f.close();
    return success();
}

struct WeightOpLowering : public OpConversionPattern<WeightOp>
{
    using OpConversionPattern<WeightOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(WeightOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        StringRef            path        = op.getPath();
        Type                 elementType = op.getElemType();
        SmallVector<int64_t> shape;
        if (auto shapeAttr = op.getShape()) {
            for (auto dim : shapeAttr.getAsRange<IntegerAttr>()) {
                shape.push_back(dim.getInt());
            }
        }

        std::vector<char> rawBytes;
        if (failed(readTensorFromFile(op.getLoc(), path, shape, rawBytes))) {
            return failure();
        }

        DenseElementsAttr denseAttr;
        auto              tensorType = RankedTensorType::get(shape, elementType);
        if (elementType.isF32()) {
            ArrayRef<float> floatData(reinterpret_cast<const float*>(rawBytes.data()),
                                      rawBytes.size() / sizeof(float));
            denseAttr = DenseElementsAttr::get(tensorType, floatData);
        }
        else {
            return emitError(op.getLoc(), "Unsupported element type in WeightOp: ") << elementType;
        }
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, denseAttr);

        return success();
    }
};

//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorSliceOp lowering
//===----------------------------------------------------------------------===//
struct TensorSliceOpLowering : public OpConversionPattern<cherry::TensorSliceOp>
{
    using OpConversionPattern<cherry::TensorSliceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(cherry::TensorSliceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        Location loc   = op.getLoc();
        Value    input = adaptor.getInput();
        auto inputType = cast<RankedTensorType>(this->typeConverter->convertType(input.getType()));
        int64_t rank   = inputType.getRank();

        auto startValues = adaptor.getStarts();
        auto sizesAttr   = op.getSizes();

        Type indexType = rewriter.getIndexType();

        SmallVector<OpFoldResult> offsets;
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides;


        for (int i = 0; i < rank; ++i) {
            Value offsetVal = startValues[i];

            Value offsetIndex = rewriter.create<arith::IndexCastOp>(loc, indexType, offsetVal);
            offsets.push_back(offsetIndex);

            auto    sizeAttr = cast<IntegerAttr>(sizesAttr[i]);
            int64_t sizeVal  = sizeAttr.getInt();
            sizes.push_back(rewriter.getIndexAttr(sizeVal));

            strides.push_back(rewriter.getIndexAttr(1));
        }

        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            op, resultType, input, offsets, sizes, strides);
        return success();
    }
};


//===----------------------------------------------------------------------===//
// ::mlir::cherry::TensorSetSliceOp lowering
//===----------------------------------------------------------------------===//
struct TensorSetSliceOpLowering : public OpConversionPattern<TensorSetSliceOp>
{
    using OpConversionPattern<TensorSetSliceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TensorSetSliceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        Location loc = op.getLoc();

        Value dest   = adaptor.getDest();
        Value source = adaptor.getSource();

        auto destType   = cast<RankedTensorType>(dest.getType());
        auto sourceType = cast<RankedTensorType>(source.getType());

        int64_t destRank   = destType.getRank();
        int64_t sourceRank = sourceType.getRank();

        auto    indices    = adaptor.getIndices();
        int64_t numIndices = indices.size();

        SmallVector<OpFoldResult> offsets;
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides;

        Type  indexType = rewriter.getIndexType();
        Value c0        = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1        = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        for (int i = 0; i < destRank; ++i) {
            if (i < numIndices) {
                Value idxVal = indices[i];
                Value idx    = rewriter.create<arith::IndexCastOp>(loc, indexType, idxVal);
                offsets.push_back(idx);
            }
            else {
                offsets.push_back(rewriter.getIndexAttr(0));
            }

            if (i < numIndices) {
                sizes.push_back(rewriter.getIndexAttr(1));
            }
            else {
                int sourceDimIdx = i - numIndices;
                if (sourceRank == destRank) {
                    sourceDimIdx = i;
                }
                sizes.push_back(rewriter.getIndexAttr(sourceType.getDimSize(sourceDimIdx)));
            }
            strides.push_back(rewriter.getIndexAttr(1));
        }
        rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
            op, source, dest, offsets, sizes, strides);

        return success();
    }
};

//===----------------------------------------------------------------------===//
// ::mlir::cherry::RopeOp lowering
//===----------------------------------------------------------------------===//
struct RopeOpLowering : public OpConversionPattern<RopeOp>
{
    using OpConversionPattern<RopeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(RopeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        Location loc   = op.getLoc();
        Value    input = adaptor.getInput();
        Value    pos   = adaptor.getPos();

        auto    inputType = cast<RankedTensorType>(input.getType());
        int64_t rank      = inputType.getRank();
        auto    shape     = inputType.getShape();
        int64_t dim       = shape[rank - 1];   // Head Dimension
        int64_t halfDim   = dim / 2;

        if (dim % 2 != 0) return op.emitError("RoPE requires last dimension to be even");

        Type elemType = inputType.getElementType();

        // Generate cos & sin Table [halfDim]
        // angle[i] = pos * 10000^(-2i/dim)
        // cosTable[i] = cos(angle[i])
        // sinTable[i] = sin(angle[i])
        Value cosTableInit =
            rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{halfDim}, elemType);
        Value sinTableInit =
            rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{halfDim}, elemType);
        Value                     posF32 = rewriter.create<arith::UIToFPOp>(loc, elemType, pos);
        SmallVector<AffineMap, 2> maps   = {rewriter.getMultiDimIdentityMap(1),
                                            rewriter.getMultiDimIdentityMap(1)};

        SmallVector<utils::IteratorType> iterators = {utils::IteratorType::parallel};

        auto tablesOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{cosTableInit.getType(), sinTableInit.getType()},
            ValueRange{},
            ValueRange{cosTableInit, sinTableInit},
            maps,
            iterators,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value i    = b.create<linalg::IndexOp>(loc, 0);
                Value iI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), i);
                Value iF32 = b.create<arith::UIToFPOp>(loc, elemType, iI64);

                Value c10000 = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemType, 10000.0));
                Value cDim =
                    b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemType, (double)dim));
                Value cMinus2 = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemType, -2.0));

                // theta = 10000 ^ (-2*i / dim)
                Value exponent_nom = b.create<arith::MulFOp>(loc, cMinus2, iF32);
                Value exponent     = b.create<arith::DivFOp>(loc, exponent_nom, cDim);
                Value theta        = b.create<math::PowFOp>(loc, c10000, exponent);

                // angle = pos * theta
                Value angle = b.create<arith::MulFOp>(loc, posF32, theta);

                Value cosVal = b.create<math::CosOp>(loc, angle);
                Value sinVal = b.create<math::SinOp>(loc, angle);

                b.create<linalg::YieldOp>(loc, ValueRange{cosVal, sinVal});
            });

        Value cosTable = tablesOp.getResult(0);
        Value sinTable = tablesOp.getResult(1);

        // expand input : 1 * 12 * 64 -> 1 * 12 * 32 * 2
        SmallVector<int64_t> expandedShape(shape.begin(), shape.end());
        expandedShape[rank - 1] = halfDim;
        expandedShape.push_back(2);
        auto expandedType = RankedTensorType::get(expandedShape, elemType);
        SmallVector<ReassociationIndices> reassoc(rank);
        for (int i = 0; i < rank - 1; ++i) reassoc[i] = {i};
        reassoc[rank - 1] = {rank - 1, rank};
        Value inputExp = rewriter.create<tensor::ExpandShapeOp>(loc, expandedType, input, reassoc);

        // Extract Even/Odd slices
        // Even: [..., :, 0] 1 * 12 * 32 * 1,
        // Odd: [..., :, 1] 1 * 12 * 32 * 1
        SmallVector<OpFoldResult> offsets(rank + 1, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides(rank + 1, rewriter.getIndexAttr(1));
        for (int i = 0; i < rank; ++i) {
            sizes.push_back(rewriter.getIndexAttr(expandedType.getDimSize(i)));
        }
        sizes.push_back(rewriter.getIndexAttr(1));

        // Extract Even
        offsets.back() = rewriter.getIndexAttr(0);
        Value evenSlice =
            rewriter.create<tensor::ExtractSliceOp>(loc, inputExp, offsets, sizes, strides);

        SmallVector<ReassociationIndices> collapseReassoc(rank);
        for (int i = 0; i < rank; ++i) collapseReassoc[i] = {i};
        collapseReassoc[rank - 1].push_back(rank);   // Merge last two

        // 1 * 12 * 32
        Value evenFlat = rewriter.create<tensor::CollapseShapeOp>(loc, evenSlice, collapseReassoc);

        offsets.back() = rewriter.getIndexAttr(1);
        Value oddSlice =
            rewriter.create<tensor::ExtractSliceOp>(loc, inputExp, offsets, sizes, strides);
        // 1 * 12 * 32
        Value oddFlat = rewriter.create<tensor::CollapseShapeOp>(loc, oddSlice, collapseReassoc);


        // Apply Rotation
        // Inputs: evenFlat [1 * 12 * 32], oddFlat [1 * 12 * 32], cos [32], sin [32]
        // Output: resEven [1 * 12 * 32], resOdd [1 * 12 * 32]

        // [1 * 12 * 32]
        Value resInit = rewriter.create<tensor::EmptyOp>(
            loc, cast<RankedTensorType>(evenFlat.getType()).getShape(), elemType);

        // 3
        int64_t   flatRank    = rank;
        AffineMap identityMap = rewriter.getMultiDimIdentityMap(flatRank);
        AffineMap broadcastMap =
            AffineMap::get(flatRank, 0, rewriter.getAffineDimExpr(flatRank - 1), op.getContext());

        SmallVector<AffineMap, 6> rotateMaps = {
            identityMap,    // even
            identityMap,    // odd
            broadcastMap,   // cos [32]
            broadcastMap,   // sin [32]
            identityMap,    // out_even
            identityMap     // out_odd
        };
        SmallVector<utils::IteratorType> rotateIterators(flatRank, utils::IteratorType::parallel);

        auto rotateOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{resInit.getType(), resInit.getType()},   // 2 outputs
            ValueRange{evenFlat, oddFlat, cosTable, sinTable},
            ValueRange{resInit, resInit},
            rotateMaps,
            rotateIterators,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value even = args[0];
                Value odd  = args[1];
                Value cos  = args[2];
                Value sin  = args[3];

                // out_even = e * c - o * s
                Value evenMulCos = b.create<arith::MulFOp>(loc, even, cos);
                Value oddMulSin  = b.create<arith::MulFOp>(loc, odd, sin);
                Value outEven    = b.create<arith::SubFOp>(loc, evenMulCos, oddMulSin);

                // out_odd = o * c + e * s
                Value oddMulCos  = b.create<arith::MulFOp>(loc, odd, cos);
                Value evenMulSin = b.create<arith::MulFOp>(loc, even, sin);
                Value outOdd     = b.create<arith::AddFOp>(loc, oddMulCos, evenMulSin);

                b.create<linalg::YieldOp>(loc, ValueRange{outEven, outOdd});
            });
        // 1 * 12 *32
        Value resEven = rotateOp.getResult(0);
        Value resOdd  = rotateOp.getResult(1);

        // Merge
        Value resEvenExp = rewriter.create<tensor::ExpandShapeOp>(
            loc, evenSlice.getType(), resEven, collapseReassoc);
        Value resOddExp = rewriter.create<tensor::ExpandShapeOp>(
            loc, oddSlice.getType(), resOdd, collapseReassoc);

        Value resultInit = rewriter.create<tensor::EmptyOp>(loc, expandedType.getShape(), elemType);

        offsets.back() = rewriter.getIndexAttr(0);
        Value merged1  = rewriter.create<tensor::InsertSliceOp>(
            loc, resEvenExp, resultInit, offsets, sizes, strides);

        offsets.back() = rewriter.getIndexAttr(1);
        Value merged2  = rewriter.create<tensor::InsertSliceOp>(
            loc, resOddExp, merged1, offsets, sizes, strides);

        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, merged2, reassoc);

        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::TensorGetOp lowering
// ===----------------------------------------------------------------------===//
struct TensorGetOpLowering : public OpConversionPattern<TensorGetOp>
{
    using OpConversionPattern<TensorGetOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TensorGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto               loc     = op.getLoc();
        auto               indices = adaptor.getIndices();
        SmallVector<Value> indicesIndex;
        for (auto indice : indices) {
            Value indiceIndex =
                rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), indice);
            indicesIndex.push_back(indiceIndex);
        }
        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, adaptor.getInput(), indicesIndex);
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::TensorSetOp lowering
// ===----------------------------------------------------------------------===//
struct TensorSetOpLowering : public OpConversionPattern<TensorSetOp>
{
    using OpConversionPattern<TensorSetOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TensorSetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto               loc     = op.getLoc();
        auto               indices = adaptor.getIndices();
        SmallVector<Value> indicesIndex;
        for (auto indice : indices) {
            Value indiceIndex =
                rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), indice);
            indicesIndex.push_back(indiceIndex);
        }
        rewriter.replaceOpWithNewOp<tensor::InsertOp>(op,
                                                      adaptor.getValue(),   // scalar to insert
                                                      adaptor.getInput(),   // dest tensor
                                                      indicesIndex          // indices
        );
        return success();
    }
};


// ===----------------------------------------------------------------------===//
// Generic Binary Scalar Op Lowering Template
// ScalarAddOp, ScalarSubOp, ScalarMulOp, ScalarDivOp
// ===----------------------------------------------------------------------===//
template<typename SourceOp, typename FloatOp, typename IntOp>
struct ScalarBinaryOpLowering : public OpConversionPattern<SourceOp>
{
    using OpConversionPattern<SourceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto  lhs  = op.getLhs();
        auto  rhs  = op.getRhs();
        auto  type = lhs.getType();
        Value result;
        if (isa<FloatType>(type)) {
            result = rewriter.create<FloatOp>(op.getLoc(), lhs, rhs);
        }
        else if (isa<IntegerType>(type)) {
            result = rewriter.create<IntOp>(op.getLoc(), lhs, rhs);
        }
        rewriter.replaceOp(op, result);
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// Generic Binary Tensor Op Lowering Template
// TensorAddOp, TensorSubOp, TensorMulOp, TensorDivOp
// ===----------------------------------------------------------------------===//
template<typename SourceOp, typename TargetLinalgOp>
struct TensorBinaryOpLowering : public OpConversionPattern<SourceOp>
{
    using OpConversionPattern<SourceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultType.getElementType());

        rewriter.replaceOpWithNewOp<TargetLinalgOp>(op,
                                                    TypeRange{resultType},
                                                    ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                                                    ValueRange{initTensor});

        return success();
    }
};

// ===----------------------------------------------------------------------===//
// Generic Unary Tensor Op Lowering Template
// TensorNegOp, TensorExpOp, TensorTanhOp
// ===----------------------------------------------------------------------===//
template<typename SourceOp, typename TargetOp>
struct TensorUnaryOpLowering : public OpConversionPattern<SourceOp>
{
    using OpConversionPattern<SourceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultType.getElementType());
        auto result = rewriter.create<TargetOp>(op.getLoc(), adaptor.getOperands()[0], initTensor);
        rewriter.replaceOp(op, result);
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::TensorReluOp lowering
// ===----------------------------------------------------------------------===//
struct TensorReluOpLowering : public OpConversionPattern<TensorReluOp>
{
    using OpConversionPattern<TensorReluOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TensorReluOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        auto inputType = cast<RankedTensorType>(adaptor.getOperand().getType());

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultType.getElementType());

        SmallVector<AffineMap, 2> indexingMaps = {
            rewriter.getMultiDimIdentityMap(inputType.getRank()),
            rewriter.getMultiDimIdentityMap(resultType.getRank())};
        SmallVector<utils::IteratorType> iteratorTypes(resultType.getRank(),
                                                       utils::IteratorType::parallel);
        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            TypeRange{resultType},
            ValueRange{adaptor.getOperand()},
            ValueRange{initTensor},
            indexingMaps,
            iteratorTypes,

            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value zero = builder.create<arith::ConstantOp>(
                    loc, builder.getFloatAttr(resultType.getElementType(), 0.0));
                Value result = builder.create<arith::MaximumFOp>(loc, args[0], zero);
                builder.create<linalg::YieldOp>(loc, result);
            });
        return success();
    }
};
// ===----------------------------------------------------------------------===//
// mlir::cherry::TensorSiluOp lowering
// ===----------------------------------------------------------------------===//
struct TensorSiluOpLowering : public OpConversionPattern<TensorSiluOp>
{
    using OpConversionPattern<TensorSiluOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TensorSiluOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        auto inputType = cast<RankedTensorType>(adaptor.getOperand().getType());

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultType.getElementType());

        SmallVector<AffineMap, 2> indexingMaps = {
            rewriter.getMultiDimIdentityMap(inputType.getRank()),
            rewriter.getMultiDimIdentityMap(resultType.getRank())};
        SmallVector<utils::IteratorType> iteratorTypes(resultType.getRank(),
                                                       utils::IteratorType::parallel);
        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            TypeRange{resultType},
            ValueRange{adaptor.getOperand()},
            ValueRange{initTensor},
            indexingMaps,
            iteratorTypes,

            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value neg = builder.create<arith::NegFOp>(loc, args[0]);
                Value exp = builder.create<math::ExpOp>(loc, neg);
                Value den = builder.create<arith::AddFOp>(loc, args[0], exp);
                Value res = builder.create<arith::DivFOp>(loc, args[0], den);
                builder.create<linalg::YieldOp>(loc, res);
            });
        return success();
    }
};
// ===----------------------------------------------------------------------===//
// mlir::cherry::TensorSigmoidOp lowering
// ===----------------------------------------------------------------------===//
struct TensorSigmoidOpLowering : public OpConversionPattern<TensorSigmoidOp>
{
    using OpConversionPattern<TensorSigmoidOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TensorSigmoidOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        auto inputType = cast<RankedTensorType>(adaptor.getOperand().getType());

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultType.getElementType());

        SmallVector<AffineMap, 2> indexingMaps = {
            rewriter.getMultiDimIdentityMap(inputType.getRank()),
            rewriter.getMultiDimIdentityMap(resultType.getRank())};
        SmallVector<utils::IteratorType> iteratorTypes(resultType.getRank(),
                                                       utils::IteratorType::parallel);
        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            TypeRange{resultType},
            ValueRange{adaptor.getOperand()},
            ValueRange{initTensor},
            indexingMaps,
            iteratorTypes,

            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value one = builder.create<arith::ConstantOp>(
                    loc, builder.getFloatAttr(resultType.getElementType(), 1.0));
                Value neg = builder.create<arith::NegFOp>(loc, args[0]);
                Value exp = builder.create<math::ExpOp>(loc, neg);
                Value den = builder.create<arith::AddFOp>(loc, one, exp);
                Value res = builder.create<arith::DivFOp>(loc, one, den);
                builder.create<linalg::YieldOp>(loc, res);
            });
        return success();
    }
};

// ============================================================================
// TensorScalarOp
// ============================================================================
template<typename OpType, typename FloatOp, typename IntOp>
struct TensorScalarOpLowering : public OpConversionPattern<OpType>
{

    using OpConversionPattern<OpType>::OpConversionPattern;

    LogicalResult matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto  loc    = op.getLoc();
        Value input  = adaptor.getInput();
        Value scalar = adaptor.getScalar();

        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        auto    inputType = cast<RankedTensorType>(input.getType());
        int64_t rank      = inputType.getRank();

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            loc, resultType.getShape(), resultType.getElementType());

        SmallVector<AffineMap, 2> indexingMaps = {
            rewriter.getMultiDimIdentityMap(rank),   // Input
            rewriter.getMultiDimIdentityMap(rank)    // Output
        };
        SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            TypeRange{resultType},
            ValueRange{input},        // Input Tensor
            ValueRange{initTensor},   // Output Tensor
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value inElem = args[0];
                Type  type   = scalar.getType();
                Value res;
                if (isa<FloatType>(type)) {
                    res = b.create<FloatOp>(loc, inElem, scalar);
                }
                else {
                    res = b.create<IntOp>(loc, inElem, scalar);
                }
                b.create<linalg::YieldOp>(loc, res);
            });

        return success();
    }
};


// ===----------------------------------------------------------------------===//
// mlir::cherry::ArgMaxOp lowering
// ===----------------------------------------------------------------------===//
struct ArgMaxOpLowering : public OpConversionPattern<ArgMaxOp>
{
    using OpConversionPattern<ArgMaxOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ArgMaxOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        Location loc       = op.getLoc();
        Value    input     = adaptor.getInput();
        auto     inputType = cast<RankedTensorType>(input.getType());
        auto     resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        int64_t rank = inputType.getRank();
        int64_t dim  = op.getDim();


        // Init Max Val
        Value minInf;
        Type  elemType = inputType.getElementType();
        minInf         = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

        Value emptyVal   = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
        Value initMaxVal = rewriter.create<linalg::FillOp>(loc, minInf, emptyVal).getResult(0);

        // Init Max Idx
        Value zeroI64 = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
        Value emptyIdx =
            rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), rewriter.getI64Type());
        Value initMaxIdx = rewriter.create<linalg::FillOp>(loc, zeroI64, emptyIdx).getResult(0);

        SmallVector<AffineExpr> inputExprs;
        SmallVector<AffineExpr> outputExprs;

        for (int i = 0; i < rank; ++i) {
            inputExprs.push_back(rewriter.getAffineDimExpr(i));
            if (i != dim) {
                outputExprs.push_back(rewriter.getAffineDimExpr(i));
            }
        }

        auto inputMap  = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
        auto outputMap = AffineMap::get(rank, 0, outputExprs, rewriter.getContext());

        SmallVector<AffineMap> indexingMaps = {
            inputMap, outputMap, outputMap};   // Input, OutVal, OutIdx

        SmallVector<utils::IteratorType> iteratorTypes;
        for (int i = 0; i < rank; ++i) {
            if (i == dim)
                iteratorTypes.push_back(utils::IteratorType::reduction);
            else
                iteratorTypes.push_back(utils::IteratorType::parallel);
        }

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{initMaxVal.getType(), initMaxIdx.getType()},   // Result types
            ValueRange{input},                                       // Inputs
            ValueRange{initMaxVal, initMaxIdx},                      // Outputs (Inits)
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value inVal  = args[0];
                Value accVal = args[1];
                Value accIdx = args[2];

                // curr Reduction dim index
                Value currentIdx    = b.create<linalg::IndexOp>(loc, dim);
                Value currentIdxI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), currentIdx);

                // inVal > accVal ?
                Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, inVal, accVal);
                // Max Value
                Value newMaxVal = b.create<arith::SelectOp>(loc, cmp, inVal, accVal);
                // Max Index
                Value newMaxIdx = b.create<arith::SelectOp>(loc, cmp, currentIdxI64, accIdx);

                b.create<linalg::YieldOp>(loc, ValueRange{newMaxVal, newMaxIdx});
            });

        rewriter.replaceOp(op, genericOp.getResult(1));

        return success();
    }
};


// ===----------------------------------------------------------------------===//
// mlir::cherry::TransposeOp lowering
// ===----------------------------------------------------------------------===//
struct TransposeOpLowering : public OpConversionPattern<cherry::TransposeOp>
{
    using OpConversionPattern<cherry::TransposeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(cherry::TransposeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        Value initTensor = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultType.getElementType());

        ArrayAttr            permAttr = op.getPermutation();
        SmallVector<int64_t> permValues;
        for (auto attr : permAttr) {
            permValues.push_back(cast<IntegerAttr>(attr).getInt());
        }
        rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
            op, adaptor.getInput(), initTensor, rewriter.getDenseI64ArrayAttr(permValues));

        return success();
    }
};


// ===----------------------------------------------------------------------===//
// mlir::cherry::ReshapeOp lowering
// ===----------------------------------------------------------------------===//
struct ReshapeOpLowering : public OpConversionPattern<ReshapeOp>
{
    using OpConversionPattern<ReshapeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(typeConverter->convertType(op.getResult().getType()));

        SmallVector<Value>   dynamicSizes;
        SmallVector<int64_t> staticSizes;

        SmallVector<Value> shapeValues;
        for (int64_t dim : resultType.getShape()) {
            Value dimVal =
                rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI64IntegerAttr(dim));
            shapeValues.push_back(dimVal);
        }

        Value shapeTensor = rewriter.create<tensor::FromElementsOp>(op.getLoc(), shapeValues);

        rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
            op, resultType, adaptor.getInput(), shapeTensor);

        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::GenerateMaskOp lowering
// ===----------------------------------------------------------------------===//
struct GenerateMaskOpLowering : public OpConversionPattern<GenerateMaskOp>
{
    using OpConversionPattern<GenerateMaskOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(GenerateMaskOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto  loc      = op.getLoc();
        Value boundary = adaptor.getBoundary();

        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        auto elemType = resultType.getElementType();

        ArrayAttr            shapeAttr = op.getShape();
        SmallVector<int64_t> staticShape;
        for (auto attr : shapeAttr) {
            staticShape.push_back(cast<IntegerAttr>(attr).getInt());
        }

        int64_t rank = staticShape.size();

        Value initTensor = rewriter.create<tensor::EmptyOp>(loc, staticShape, elemType);

        SmallVector<AffineMap, 1> indexingMaps = {rewriter.getMultiDimIdentityMap(rank)};

        SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            TypeRange{resultType},
            ValueRange{},             // inputs: none
            ValueRange{initTensor},   // outputs
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                int64_t lastDim = rank - 1;
                Value   idx     = b.create<linalg::IndexOp>(loc, lastDim);
                Value   idxI64  = b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);

                // idx <= boundary
                Value cond =
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, idxI64, boundary);

                Value one  = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemType, 1.0));
                Value zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemType, 0.0));
                Value res  = b.create<arith::SelectOp>(loc, cond, one, zero);

                b.create<linalg::YieldOp>(loc, res);
            });

        return success();
    }
};


// ===----------------------------------------------------------------------===//
// mlir::cherry::MaskedMatMulOp lowering
// ===----------------------------------------------------------------------===//
struct MaskedMatMulOpLowering : public OpConversionPattern<MaskedMatMulOp>
{
    using OpConversionPattern<MaskedMatMulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MaskedMatMulOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto  loc  = op.getLoc();
        Value lhs  = adaptor.getLhs();
        Value rhs  = adaptor.getRhs();
        Value mask = adaptor.getMask();

        auto lhsType  = cast<RankedTensorType>(lhs.getType());
        auto rhsType  = cast<RankedTensorType>(rhs.getType());
        auto maskType = cast<RankedTensorType>(mask.getType());
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        auto elemType = resultType.getElementType();

        int64_t lhsRank = lhsType.getRank();
        int64_t rhsRank = rhsType.getRank();
        int64_t outRank = resultType.getRank();

        int64_t lhsBatchRank  = lhsRank - 2;
        int64_t rhsBatchRank  = rhsRank - 2;
        int64_t outBatchRank  = outRank - 2;
        int64_t totalLoopRank = outBatchRank + 3;

        int64_t idxM = totalLoopRank - 3;   // index of M
        int64_t idxN = totalLoopRank - 2;   // index of N
        int64_t idxK = totalLoopRank - 1;   // index of K

        SmallVector<AffineExpr> lhsExprs;
        SmallVector<AffineExpr> rhsExprs;
        SmallVector<AffineExpr> maskExprs;   // Mask Map
        SmallVector<AffineExpr> outExprs;

        int64_t lhsBatchOffset = outBatchRank - lhsBatchRank;
        for (int i = 0; i < lhsBatchRank; ++i) {
            lhsExprs.push_back(rewriter.getAffineDimExpr(i + lhsBatchOffset));
        }
        lhsExprs.push_back(rewriter.getAffineDimExpr(idxM));
        lhsExprs.push_back(rewriter.getAffineDimExpr(idxK));

        int64_t rhsBatchOffset = outBatchRank - rhsBatchRank;
        for (int i = 0; i < rhsBatchRank; ++i) {
            rhsExprs.push_back(rewriter.getAffineDimExpr(i + rhsBatchOffset));
        }
        rhsExprs.push_back(rewriter.getAffineDimExpr(idxK));
        rhsExprs.push_back(rewriter.getAffineDimExpr(idxN));

        for (int i = 0; i < outBatchRank; ++i) {
            auto expr = rewriter.getAffineDimExpr(i);
            outExprs.push_back(expr);
            maskExprs.push_back(expr);
        }
        outExprs.push_back(rewriter.getAffineDimExpr(idxM));
        outExprs.push_back(rewriter.getAffineDimExpr(idxN));

        maskExprs.push_back(rewriter.getAffineDimExpr(idxM));
        maskExprs.push_back(rewriter.getAffineDimExpr(idxN));

        SmallVector<AffineMap, 4> indexingMaps = {
            AffineMap::get(totalLoopRank, 0, lhsExprs, rewriter.getContext()),
            AffineMap::get(totalLoopRank, 0, rhsExprs, rewriter.getContext()),
            AffineMap::get(totalLoopRank, 0, maskExprs, rewriter.getContext()),   // Add Mask Map
            AffineMap::get(totalLoopRank, 0, outExprs, rewriter.getContext())};

        SmallVector<utils::IteratorType> iteratorTypes(totalLoopRank,
                                                       utils::IteratorType::parallel);
        iteratorTypes[idxK] = utils::IteratorType::reduction;

        // init Output Tensor with 0.0
        Value initTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);

        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));

        Value zeroInit = rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            TypeRange{resultType},
            ValueRange{lhs, rhs, mask},   // Inputs: LHS, RHS, Mask
            ValueRange{zeroInit},         // Outputs
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value l   = args[0];
                Value r   = args[1];
                Value m   = args[2];   // Mask Value
                Value acc = args[3];

                Value mul = b.create<arith::MulFOp>(loc, l, r);
                Value sum = b.create<arith::AddFOp>(loc, acc, mul);

                Value negInf = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemType, -1.0e9));

                Value threshold = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemType, 0.5));
                Value isValid =
                    b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, m, threshold);

                Value res = b.create<arith::SelectOp>(loc, isValid, sum, negInf);

                b.create<linalg::YieldOp>(loc, res);
            });

        return success();
    }
};


// ===----------------------------------------------------------------------===//
// mlir::cherry::MatMulOp lowering
// (x*y*z*m*k) X (y*z*k*n) => x*y*z*m*n
// A X B => C
// x = d0, y = d1, z = d2, m = d3, n = d4, k = d5
// #mapA = affine_map<(d0,d1,d2,d3,d4,d5) -> (d0,d1,d2,d3,d5)>
// #mapB = affine_map<(d0,d1,d2,d3,d4,d5) -> (d1,d2,d5,d4)>
// #mapC = affine_map<(d0,d1,d2,d3,d4,d5) -> (d0,d1,d2,d3,d4)>
// for d0 in 0..x:
//  for d1 in 0..y:
//   for d2 in 0..z:
//    for d3 in 0..m:
//     for d4 in 0..n:
//      sum=0
//      for d5 in 0..k:
//          val_a = A[d0,d1,d2,d3,d5]
//          val_b = B[d1,d2,d5,d4]
//          sum += val_a * val_b
//      C[d0,d1,d2,d3,d4] += sum
// ===----------------------------------------------------------------------===//
struct MatMulOpLowering : public OpConversionPattern<MatMulOp>
{
    using OpConversionPattern<MatMulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto  loc = op.getLoc();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        auto lhsType = cast<RankedTensorType>(lhs.getType());
        auto rhsType = cast<RankedTensorType>(rhs.getType());
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));

        int64_t lhsRank = lhsType.getRank();
        int64_t rhsRank = rhsType.getRank();
        int64_t outRank = resultType.getRank();

        int64_t lhsBatchRank  = lhsRank - 2;
        int64_t rhsBatchRank  = rhsRank - 2;
        int64_t outBatchRank  = outRank - 2;
        int64_t totalLoopRank = outBatchRank + 3;

        int64_t idxM = totalLoopRank - 3;   // index of M
        int64_t idxN = totalLoopRank - 2;   // index of N
        int64_t idxK = totalLoopRank - 1;   // index of K

        SmallVector<AffineExpr> lhsExprs;
        SmallVector<AffineExpr> rhsExprs;
        SmallVector<AffineExpr> outExprs;

        int64_t lhsBatchOffset = outBatchRank - lhsBatchRank;
        for (int i = 0; i < lhsBatchRank; ++i) {
            lhsExprs.push_back(rewriter.getAffineDimExpr(i + lhsBatchOffset));
        }
        lhsExprs.push_back(rewriter.getAffineDimExpr(idxM));   // M
        lhsExprs.push_back(rewriter.getAffineDimExpr(idxK));   // K

        int64_t rhsBatchOffset = outBatchRank - rhsBatchRank;
        for (int i = 0; i < rhsBatchRank; ++i) {
            rhsExprs.push_back(rewriter.getAffineDimExpr(i + rhsBatchOffset));
        }
        rhsExprs.push_back(rewriter.getAffineDimExpr(idxK));   // K
        rhsExprs.push_back(rewriter.getAffineDimExpr(idxN));   // N

        for (int i = 0; i < outBatchRank; ++i) {
            outExprs.push_back(rewriter.getAffineDimExpr(i));
        }
        outExprs.push_back(rewriter.getAffineDimExpr(idxM));   // M
        outExprs.push_back(rewriter.getAffineDimExpr(idxN));   // N

        SmallVector<AffineMap, 3> indexingMaps = {
            AffineMap::get(totalLoopRank, 0, lhsExprs, rewriter.getContext()),
            AffineMap::get(totalLoopRank, 0, rhsExprs, rewriter.getContext()),
            AffineMap::get(totalLoopRank, 0, outExprs, rewriter.getContext())};

        // Parallel : Batch + M + N, Reduction : K
        SmallVector<utils::IteratorType> iteratorTypes(totalLoopRank,
                                                       utils::IteratorType::parallel);
        iteratorTypes[idxK] = utils::IteratorType::reduction;

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            loc, resultType.getShape(), resultType.getElementType());

        Value zero = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(resultType.getElementType(), 0.0));

        Value zeroInit = rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            TypeRange{resultType},
            ValueRange{lhs, rhs},
            ValueRange{zeroInit},
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value l   = args[0];
                Value r   = args[1];
                Value acc = args[2];

                Value mul = b.create<arith::MulFOp>(loc, l, r);
                Value res = b.create<arith::AddFOp>(loc, acc, mul);

                b.create<linalg::YieldOp>(loc, res);
            });

        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::SoftmaxOp lowering
// ===----------------------------------------------------------------------===//
struct SoftmaxOpLowering : public OpConversionPattern<SoftmaxOp>
{
    using OpConversionPattern<SoftmaxOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        // auto resultType =
        //     cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));

        // Value initTensor = rewriter.create<tensor::EmptyOp>(
        //     op.getLoc(), resultType.getShape(), resultType.getElementType());
        // rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(
        //     op, TypeRange{resultType}, adaptor.getInput(), initTensor, op.getAxisAttr());
        Location loc   = op.getLoc();
        Value    input = adaptor.getInput();
        int64_t  axis  = op.getAxis();

        auto inputType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getInput().getType()));
        auto    elemType = inputType.getElementType();
        int64_t rank     = inputType.getRank();

        SmallVector<AffineExpr>          fullExprs;
        SmallVector<AffineExpr>          reductionExprs;
        SmallVector<utils::IteratorType> reductionIters;
        SmallVector<utils::IteratorType> parallelIters;

        for (int i = 0; i < rank; i++) {
            fullExprs.push_back(rewriter.getAffineDimExpr(i));
            parallelIters.push_back(utils::IteratorType::parallel);
            if (i == axis) {
                reductionIters.push_back(utils::IteratorType::reduction);
            }
            else {
                reductionIters.push_back(utils::IteratorType::parallel);
                reductionExprs.push_back(rewriter.getAffineDimExpr(i));
            }
        }
        auto fullMap      = AffineMap::get(rank, 0, fullExprs, rewriter.getContext());
        auto reductionMap = AffineMap::get(rank, 0, reductionExprs, rewriter.getContext());

        // ---------------------------------------------------------
        //  Max
        // ---------------------------------------------------------
        SmallVector<OpFoldResult> inputDims = tensor::getMixedSizes(rewriter, loc, input);
        SmallVector<OpFoldResult> reducedDims;
        for (int i = 0; i < rank; ++i) {
            if (i != axis) reducedDims.push_back(inputDims[i]);
        }

        Value initMax = rewriter.create<tensor::EmptyOp>(loc, reducedDims, elemType);
        Value negInf  = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<double>::infinity()));
        Value filledMax = rewriter.create<linalg::FillOp>(loc, negInf, initMax).getResult(0);

        auto maxOp = rewriter.create<linalg::GenericOp>(
            loc,
            filledMax.getType(),
            ValueRange{input},
            ValueRange{filledMax},
            ArrayRef<AffineMap>{fullMap, reductionMap},
            reductionIters,
            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value val = builder.create<arith::MaxNumFOp>(loc, args[0], args[1]);
                builder.create<linalg::YieldOp>(loc, val);
            });
        Value maxVal = maxOp.getResult(0);


        // ---------------------------------------------------------
        //  Exp(Input - Max) (Element-wise)
        // ---------------------------------------------------------
        Value initExp = rewriter.create<tensor::EmptyOp>(loc, inputDims, elemType);

        auto expOp = rewriter.create<linalg::GenericOp>(
            loc,
            initExp.getType(),
            ValueRange{input, maxVal},
            ValueRange{initExp},
            ArrayRef<AffineMap>{fullMap, reductionMap, fullMap},
            parallelIters,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value in  = args[0];
                Value max = args[1];
                Value sub = b.create<arith::SubFOp>(loc, in, max);
                Value exp = b.create<math::ExpOp>(loc, sub);
                b.create<linalg::YieldOp>(loc, exp);
            });
        Value expVal = expOp.getResult(0);

        // ---------------------------------------------------------
        // Sum (Reduction)
        // ---------------------------------------------------------
        Value initSum = rewriter.create<tensor::EmptyOp>(loc, reducedDims, elemType);
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
        Value filledSum = rewriter.create<linalg::FillOp>(loc, zero, initSum).getResult(0);

        auto sumOp = rewriter.create<linalg::GenericOp>(
            loc,
            filledSum.getType(),
            ValueRange{expVal},
            ValueRange{filledSum},
            ArrayRef<AffineMap>{fullMap, reductionMap},
            parallelIters,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value val = b.create<arith::AddFOp>(loc, args[0], args[1]);
                b.create<linalg::YieldOp>(loc, val);
            });
        Value sumVal = sumOp.getResult(0);

        // ---------------------------------------------------------
        // Div (Exp / Sum) (Element-wise)
        // ---------------------------------------------------------
        Value initResult = rewriter.create<tensor::EmptyOp>(loc, inputDims, elemType);

        auto divOp = rewriter.create<linalg::GenericOp>(
            loc,
            initResult.getType(),
            ValueRange{expVal, sumVal},
            ValueRange{initResult},
            ArrayRef<AffineMap>{fullMap, reductionMap, fullMap},
            parallelIters,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value exp = args[0];
                Value sum = args[1];
                Value div = b.create<arith::DivFOp>(loc, exp, sum);
                b.create<linalg::YieldOp>(loc, div);
            });

        rewriter.replaceOp(op, divOp.getResult(0));
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::BroadcastOp lowering
// ===----------------------------------------------------------------------===//
struct BroadcastOpLowering : public OpConversionPattern<BroadcastOp>
{
    using OpConversionPattern<BroadcastOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        auto inputType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getInput().getType()));

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultType.getElementType());
        SmallVector<int64_t> dims;
        for (int64_t i = 0; i < resultType.getRank() - inputType.getRank(); i++) {
            dims.push_back(i + inputType.getRank());
        }
        rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(
            op, adaptor.getInput(), initTensor, rewriter.getDenseI64ArrayAttr(dims));
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::RMSNormOp lowering
// ===----------------------------------------------------------------------===//
struct RMSNormOpLowering : public OpConversionPattern<RMSNormOp>
{
    using OpConversionPattern<RMSNormOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(RMSNormOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto loc = op.getLoc();

        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        auto inputType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getInput().getType()));

        auto input    = adaptor.getInput();
        auto scale    = adaptor.getScale();
        auto eps      = adaptor.getEpsilon();
        auto rank     = inputType.getRank();
        auto elemType = inputType.getElementType();

        SmallVector<AffineExpr>          fullExprs;
        SmallVector<AffineExpr>          reductionExprs;
        SmallVector<utils::IteratorType> reductionIterators;
        SmallVector<utils::IteratorType> parallelIterators;

        for (int i = 0; i < rank; i++) {
            fullExprs.push_back(rewriter.getAffineDimExpr(i));
            parallelIterators.push_back(utils::IteratorType::parallel);
            if (i == rank - 1) {
                reductionIterators.push_back(utils::IteratorType::reduction);
            }
            else {
                reductionExprs.push_back(rewriter.getAffineDimExpr(i));
                reductionIterators.push_back(utils::IteratorType::parallel);
            }
        }
        auto fullMap      = AffineMap::get(rank, 0, fullExprs, rewriter.getContext());
        auto reductionMap = AffineMap::get(rank, 0, reductionExprs, rewriter.getContext());
        auto scaleMap     = AffineMap::get(rank, 0, fullExprs.back(), rewriter.getContext());

        SmallVector<OpFoldResult> inputDims = tensor::getMixedSizes(rewriter, loc, input);
        SmallVector<OpFoldResult> reductionDims;
        for (int i = 0; i < rank - 1; i++) {
            reductionDims.push_back(inputDims[i]);
        }

        // ---------------------------------------------------------
        // Sum(x ^ 2)
        // ---------------------------------------------------------
        Value initSum = rewriter.create<tensor::EmptyOp>(loc, reductionDims, elemType);
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
        Value filledSum = rewriter.create<linalg::FillOp>(loc, zero, initSum).getResult(0);

        auto sumSqOp = rewriter.create<linalg::GenericOp>(
            loc,
            filledSum.getType(),
            ValueRange{input},
            ValueRange{filledSum},
            ArrayRef<AffineMap>{fullMap, reductionMap},
            reductionIterators,
            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value sum    = args[1];
                Value x      = args[0];
                Value xSq    = builder.create<arith::MulFOp>(loc, x, x);
                Value newSum = builder.create<arith::AddFOp>(loc, sum, xSq);
                builder.create<linalg::YieldOp>(loc, newSum);
            });
        auto sumSq = sumSqOp.getResult(0);


        // ---------------------------------------------------------
        // rsqrt =  1 / sqrt( sumSq / N + eps )
        // ---------------------------------------------------------
        Value lastDim    = rewriter.create<tensor::DimOp>(loc, input, rank - 1);
        Value lastDimI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), lastDim);
        Value lastDimFloat = rewriter.create<arith::UIToFPOp>(loc, elemType, lastDimI64);
        Value epsVal =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, eps));

        SmallVector<AffineExpr>          rsqrtExprs;
        SmallVector<utils::IteratorType> rsqrtIterators;

        for (int i = 0; i < rank - 1; i++) {
            rsqrtExprs.push_back(rewriter.getAffineDimExpr(i));
            rsqrtIterators.push_back(utils::IteratorType::parallel);
        }
        auto rsqrtMap = AffineMap::get(rank - 1, 0, rsqrtExprs, rewriter.getContext());

        Value initRsqrt = rewriter.create<tensor::EmptyOp>(loc, reductionDims, elemType);
        auto  rsqrtOp   = rewriter.create<linalg::GenericOp>(
            loc,
            initRsqrt.getType(),
            ValueRange{sumSq},
            ValueRange{initRsqrt},
            ArrayRef<AffineMap>{rsqrtMap, rsqrtMap},
            rsqrtIterators,
            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value sumSqVal    = args[0];
                Value mean        = builder.create<arith::DivFOp>(loc, sumSqVal, lastDimFloat);
                Value meanPlusEps = builder.create<arith::AddFOp>(loc, mean, epsVal);
                Value rsqrt       = builder.create<math::RsqrtOp>(loc, meanPlusEps);
                builder.create<linalg::YieldOp>(loc, rsqrt);
            });
        auto rsqrt = rsqrtOp.getResult(0);

        // ---------------------------------------------------------
        // result = input * rsqrt * scale
        // ---------------------------------------------------------
        Value initResult = rewriter.create<tensor::EmptyOp>(loc, inputDims, elemType);
        auto  finalOp    = rewriter.create<linalg::GenericOp>(
            loc,
            initResult.getType(),
            ValueRange{input, rsqrt, scale},
            ValueRange{initResult},
            ArrayRef<AffineMap>{fullMap, reductionMap, scaleMap, fullMap},
            parallelIterators,
            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value input      = args[0];
                Value rsqrt      = args[1];
                Value scale      = args[2];
                Value normalized = builder.create<arith::MulFOp>(loc, input, rsqrt);
                Value result     = builder.create<arith::MulFOp>(loc, normalized, scale);
                builder.create<linalg::YieldOp>(loc, result);
            });
        rewriter.replaceOp(op, finalOp.getResult(0));
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::CastOp Lowering
// ===----------------------------------------------------------------------===//
struct CastOpLowering : public OpConversionPattern<CastOp>
{
    using OpConversionPattern<CastOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto resultType =
            cast<RankedTensorType>(this->typeConverter->convertType(op.getResult().getType()));
        rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, adaptor.getInput());
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::CallOp Lowering
// ===----------------------------------------------------------------------===//
struct CallOpLowering : public OpConversionPattern<CallOp>
{
    using OpConversionPattern<CallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto              loc = op.getLoc();
        SmallVector<Type> returnTypes;
        for (auto result : op.getResults()) {
            returnTypes.push_back(
                cast<RankedTensorType>(this->typeConverter->convertType(result.getType())));
        }
        rewriter.replaceOpWithNewOp<func::CallOp>(
            op, op.getCalleeAttr(), returnTypes, adaptor.getInputs());
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::ReturnOp Lowering
// ===----------------------------------------------------------------------===//
struct ReturnOpLowering : public OpConversionPattern<ReturnOp>
{
    using OpConversionPattern<ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::FuncOp Lowering
// ===----------------------------------------------------------------------===//
struct FuncOpLowering : public OpConversionPattern<FuncOp>
{
    using OpConversionPattern<FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        auto                               type = op.getFunctionType();
        TypeConverter::SignatureConversion result(type.getNumInputs());

        for (auto [index, inputType] : llvm::enumerate(type.getInputs())) {
            if (failed(typeConverter->convertSignatureArg(index, inputType, result)))
                return failure();
        }

        SmallVector<Type, 1> newResultTypes;
        if (failed(typeConverter->convertTypes(type.getResults(), newResultTypes)))
            return failure();

        auto newFuncOp = rewriter.create<func::FuncOp>(
            op.getLoc(),
            op.getSymName(),
            rewriter.getFunctionType(result.getConvertedTypes(), newResultTypes));

        rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.end());

        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter, &result)))
            return failure();

        rewriter.eraseOp(op);
        return success();
    }
};

// ===----------------------------------------------------------------------===//
// mlir::cherry::PrintOp Lowering
// ===----------------------------------------------------------------------===//
struct PrintOpLowering : public OpConversionPattern<PrintOp>
{
    using OpConversionPattern<PrintOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override
    {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        Value    input  = adaptor.getInput();

        auto tensorType  = cast<RankedTensorType>(input.getType());
        Type elementType = tensorType.getElementType();

        auto unrankedTensorType = UnrankedTensorType::get(tensorType.getElementType());

        std::string funcName;
        if (elementType.isF32()) {
            funcName = "printMemrefF32";
        }
        else if (elementType.isF64()) {
            funcName = "printMemrefF64";
        }
        else if (elementType.isInteger(32)) {
            funcName = "printMemrefI32";
        }
        else if (elementType.isInteger(64)) {
            funcName = "printMemrefI64";
        }
        if (funcName.empty()) {
            return rewriter.notifyMatchFailure(op, "Unsupported element type for print operation");
        }

        if (!module.lookupSymbol<func::FuncOp>(funcName)) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());

            auto funcType = rewriter.getFunctionType({unrankedTensorType}, {});
            auto funcOp   = rewriter.create<func::FuncOp>(module.getLoc(), funcName, funcType);
            funcOp.setPrivate();

            funcOp.setArgAttr(0, "bufferization.access", rewriter.getStringAttr("read"));
        }

        Value unrankedInput =
            rewriter.create<tensor::CastOp>(op.getLoc(), unrankedTensorType, input);

        rewriter.replaceOpWithNewOp<func::CallOp>(
            op, funcName, TypeRange{}, ValueRange{unrankedInput});

        return success();
    }
};


struct ConvertCherryToLinalgPass
    : mlir::cherry::impl::ConvertCherryToLinalgPassBase<ConvertCherryToLinalgPass>
{
    using mlir::cherry::impl::ConvertCherryToLinalgPassBase<
        ConvertCherryToLinalgPass>::ConvertCherryToLinalgPassBase;
    void runOnOperation() override;
};

void ConvertCherryToLinalgPass::runOnOperation()
{
    // Implementation of the pass
    llvm::outs() << "Running Cherry to Linalg conversion pass\n";
    ModuleOp            module = getOperation();
    CherryTypeConverter converter;

    ConversionTarget target(getContext());

    target.addLegalDialect<linalg::LinalgDialect,
                           func::FuncDialect,
                           tensor::TensorDialect,
                           arith::ArithDialect,
                           math::MathDialect>();
    target.addLegalDialect<scf::SCFDialect>();   // 先设为 legal，然后下面添加动态规则覆盖它
    target.addDynamicallyLegalDialect<scf::SCFDialect>(
        [&](Operation* op) { return converter.isLegal(op); });


    target.addIllegalOp<cherry::ConstantOp>();
    target.addIllegalOp<cherry::CreateTensorOp>();
    target.addIllegalOp<cherry::WeightOp>();
    target.addIllegalOp<cherry::TensorSliceOp>();
    target.addIllegalOp<cherry::TensorSetSliceOp>();
    target.addIllegalOp<cherry::RopeOp>();
    target.addIllegalOp<cherry::TensorGetOp>();
    target.addIllegalOp<cherry::TensorSetOp>();
    target.addIllegalOp<cherry::ReturnOp>();
    target.addIllegalOp<cherry::FuncOp>();
    target.addIllegalOp<cherry::ScalarAddOp>();
    target.addIllegalOp<cherry::ScalarSubOp>();
    target.addIllegalOp<cherry::ScalarMulOp>();
    target.addIllegalOp<cherry::ScalarDivOp>();
    target.addIllegalOp<cherry::TensorAddOp>();
    target.addIllegalOp<cherry::TensorSubOp>();
    target.addIllegalOp<cherry::TensorMulOp>();
    target.addIllegalOp<cherry::TensorDivOp>();
    target.addIllegalOp<cherry::TensorNegOp>();
    target.addIllegalOp<cherry::TensorExpOp>();
    target.addIllegalOp<cherry::TensorTanhOp>();
    target.addIllegalOp<cherry::TensorReluOp>();
    target.addIllegalOp<cherry::TensorSiluOp>();
    target.addIllegalOp<cherry::TensorSigmoidOp>();
    target.addIllegalOp<cherry::ArgMaxOp>();
    target.addIllegalOp<cherry::TransposeOp>();
    target.addIllegalOp<cherry::ReshapeOp>();
    target.addIllegalOp<cherry::GenerateMaskOp>();
    target.addIllegalOp<cherry::MaskedMatMulOp>();
    target.addIllegalOp<cherry::MatMulOp>();
    target.addIllegalOp<cherry::SoftmaxOp>();
    target.addIllegalOp<cherry::BroadcastOp>();
    target.addIllegalOp<cherry::RMSNormOp>();
    target.addIllegalOp<cherry::CastOp>();
    target.addIllegalOp<cherry::CallOp>();
    target.addIllegalOp<cherry::PrintOp>();
    RewritePatternSet patterns(&getContext());


    patterns.add<ConstantOpLowering,
                 CreateTensorOpLowering,
                 WeightOpLowering,
                 TensorSliceOpLowering,
                 TensorSetSliceOpLowering,
                 RopeOpLowering,
                 TensorGetOpLowering,
                 TensorSetOpLowering,

                 ScalarBinaryOpLowering<ScalarAddOp, arith::AddFOp, arith::AddIOp>,
                 ScalarBinaryOpLowering<ScalarSubOp, arith::SubFOp, arith::SubIOp>,
                 ScalarBinaryOpLowering<ScalarMulOp, arith::MulFOp, arith::MulIOp>,
                 ScalarBinaryOpLowering<ScalarDivOp, arith::DivFOp, arith::DivSIOp>,
                 TensorBinaryOpLowering<TensorAddOp, linalg::AddOp>,
                 TensorBinaryOpLowering<TensorSubOp, linalg::SubOp>,
                 TensorBinaryOpLowering<TensorMulOp, linalg::MulOp>,
                 TensorBinaryOpLowering<TensorDivOp, linalg::DivOp>,

                 TensorUnaryOpLowering<TensorNegOp, linalg::NegFOp>,
                 TensorUnaryOpLowering<TensorExpOp, linalg::ExpOp>,
                 TensorUnaryOpLowering<TensorTanhOp, linalg::TanhOp>,
                 TensorReluOpLowering,
                 TensorSiluOpLowering,
                 TensorSigmoidOpLowering,
                 ArgMaxOpLowering,

                 TensorScalarOpLowering<cherry::TensorMulScalarOp, arith::MulFOp, arith::MulIOp>,
                 TensorScalarOpLowering<cherry::TensorDivScalarOp, arith::DivFOp, arith::DivSIOp>,
                 TensorScalarOpLowering<cherry::TensorAddScalarOp, arith::AddFOp, arith::AddIOp>,
                 TensorScalarOpLowering<cherry::TensorSubScalarOp, arith::SubFOp, arith::SubIOp>,

                 TransposeOpLowering,
                 ReshapeOpLowering,

                 GenerateMaskOpLowering,
                 MaskedMatMulOpLowering,
                 MatMulOpLowering,
                 SoftmaxOpLowering,
                 BroadcastOpLowering,
                 RMSNormOpLowering,

                 CastOpLowering,
                 CallOpLowering,
                 ReturnOpLowering,
                 FuncOpLowering,
                 PrintOpLowering>(converter, &getContext());
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
