#include "Dialect/Cherry/IR/CherryDialect.h"
#include "Dialect/Cherry/IR/CherryOps.h"
#include "Dialect/Cherry/IR/CherryTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "pybind11/pytypes.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace mlir;

#define DEFINE_TENSOR_BINARY_OP(Name, OpClass)                                \
    PyValue Name(PyValue lhs, PyValue rhs)                                    \
    {                                                                         \
        auto inputType = cast<cherry::CherryTensorType>(lhs.value.getType()); \
        auto op        = builder->create<cherry::OpClass>(                    \
            builder->getUnknownLoc(),                                  \
            this->createDynamicTensorType(inputType.getElementType()), \
            lhs.value,                                                 \
            rhs.value);                                                \
        return PyValue(op.getResult());                                       \
    }

#define DEFINE_TENSOR_UNARY_OP(Name, OpClass)                                     \
    PyValue Name(PyValue operand)                                                 \
    {                                                                             \
        auto inputType = cast<cherry::CherryTensorType>(operand.value.getType()); \
        auto op        = builder->create<cherry::OpClass>(                        \
            builder->getUnknownLoc(),                                      \
            this->createDynamicTensorType(inputType.getElementType()),     \
            operand.value);                                                \
        return PyValue(op.getResult());                                           \
    }

#define DEFINE_TENSOR_SCALAR_OP(Name, OpClass)                                  \
    PyValue Name(PyValue input, PyValue scalar)                                 \
    {                                                                           \
        auto inputType = cast<cherry::CherryTensorType>(input.value.getType()); \
        auto op        = builder->create<cherry::OpClass>(                      \
            builder->getUnknownLoc(),                                    \
            this->createDynamicTensorType(inputType.getElementType()),   \
            input.value,                                                 \
            scalar.value);                                               \
        return PyValue(op.getResult());                                         \
    }

#define DEFINE_SCALAR_OP(Name, OpClass)                                          \
    PyValue Name(PyValue lhs, PyValue rhs)                                       \
    {                                                                            \
        Type resultType = lhs.value.getType();                                   \
        auto op         = builder->create<cherry::OpClass>(                      \
            builder->getUnknownLoc(), resultType, lhs.value, rhs.value); \
        return PyValue{op.getResult()};                                          \
    }

struct PyValue
{
    Value value;
    PyValue(Value v)
        : value(v)
    {
    }
};
struct PyType
{
    mlir::Type type;
    PyType(mlir::Type t)
        : type(t)
    {
    }
    virtual ~PyType() = default;
};

struct PyCherryTensorType : public PyType
{
    PyCherryTensorType(mlir::Type t)
        : PyType(t)
    {
        if (!isa<cherry::CherryTensorType>(t)) {
            throw std::runtime_error("Type is not a CherryTensorType");
        }
    }
};

class IrGenerator
{
private:
    mlir::Type createDynamicTensorType(mlir::Type elementType = FloatType())
    {
        return mlir::cherry::CherryTensorType::get(context.get(), {-1}, elementType);
    }

public:
    IrGenerator()
    {
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
        context = std::make_unique<MLIRContext>(registry);
        context->loadAllAvailableDialects();

        builder = std::make_unique<OpBuilder>(context.get());
        module  = ModuleOp::create(builder->getUnknownLoc());
    }

    PyType createType(std::string name)
    {
        if (name == "f32") return PyType(builder->getF32Type());
        if (name == "i32") return PyType(builder->getI32Type());
        if (name == "i64") return PyType(builder->getI64Type());
        if (name == "index") return PyType(builder->getIndexType());
        throw std::runtime_error("Unknown scalar type: " + name);
    }

    PyType createType(mlir::Type type)
    {
        if (isa<mlir::FloatType>(type)) return PyType(builder->getF32Type());
        if (isa<mlir::IntegerType>(type)) return PyType(builder->getI32Type());
        if (isa<mlir::IndexType>(type)) return PyType(builder->getIndexType());
        throw std::runtime_error("Unknown type");
    }

    PyCherryTensorType createTensorType(std::vector<int64_t> shape, PyType elementType)
    {
        std::vector<int64_t> mlir_shape;
        for (auto s : shape) {
            if (s == -1)
                mlir_shape.push_back(mlir::ShapedType::kDynamic);
            else
                mlir_shape.push_back(s);
        }
        auto type =
            cherry::CherryTensorType::get(builder->getContext(), mlir_shape, elementType.type);
        return PyCherryTensorType(type);
    }

    py::list createFunction(std::string name, std::vector<PyType*> args = {},
                            std::vector<PyType*> rets = {}, bool isPrivate = false)
    {
        builder->setInsertionPointToEnd(module->getBody());
        auto              loc = builder->getUnknownLoc();
        std::vector<Type> inputTypes;
        for (auto* arg : args) inputTypes.push_back(arg->type);
        std::vector<Type> resultTypes;
        for (auto* ret : rets) resultTypes.push_back(ret->type);

        auto funcType = builder->getFunctionType(inputTypes, resultTypes);
        auto funcOp   = builder->create<cherry::FuncOp>(loc, name, funcType);

        if (isPrivate) funcOp.setPrivate();
        mlir::Block* entryBlock = &(funcOp.front());
        builder->setInsertionPointToStart(entryBlock);

        py::list pyArgs;
        for (auto arg : entryBlock->getArguments()) {
            pyArgs.append(PyValue(arg));
        }
        return pyArgs;
    }

    py::object callOp(std::string callee_name, py::list args)
    {
        auto loc = builder->getUnknownLoc();

        std::vector<Value> call_args;
        for (auto handle : args) {
            call_args.push_back(handle.cast<PyValue*>()->value);
        }

        auto calleeFunc = module->lookupSymbol<cherry::FuncOp>(callee_name);
        if (!calleeFunc) throw std::runtime_error("Function not found: " + callee_name);

        auto resultTypes = calleeFunc.getFunctionType().getResults();

        auto callOp = builder->create<cherry::CallOp>(loc, callee_name, resultTypes, call_args);

        int numResults = callOp.getNumResults();
        if (numResults == 1) {
            return py::cast(new PyValue(callOp.getResult(0)));
        }
        else {
            py::list results;
            for (auto res : callOp.getResults()) {
                results.append(new PyValue(res));
            }
            return results;
        }
        return py::none();
    }

    void runtimeCallOp(const std::string& callee, py::args args, py::kwargs kwargs)
    {

        auto               loc = builder->getUnknownLoc();
        std::vector<Value> operands;
        for (auto handle : args) {
            try {
                PyValue* val = handle.cast<PyValue*>();
                operands.push_back(val->value);
            }
            catch (const py::cast_error&) {
                throw std::runtime_error("Call arguments must be Value objects");
            }
        }

        ArrayAttr strArgsAttr;
        if (kwargs.contains("str_args")) {
            std::vector<Attribute> attrs;
            py::list               str_list = kwargs["str_args"].cast<py::list>();
            for (auto str : str_list) {
                attrs.push_back(builder->getStringAttr(str.cast<std::string>()));
            }
            strArgsAttr = builder->getArrayAttr(attrs);
        }
        builder->create<cherry::RuntimeCallOp>(
            loc, builder->getStringAttr(callee), operands, strArgsAttr);
    }


    void returnOp(py::list args)
    {
        auto loc = builder->getUnknownLoc();

        std::vector<Value> operands;
        for (auto handle : args) {
            try {
                PyValue* val = handle.cast<PyValue*>();
                operands.push_back(val->value);
            }
            catch (const py::cast_error&) {
                throw std::runtime_error("Return arguments must be Value objects");
            }
        }
        builder->create<cherry::ReturnOp>(loc, ValueRange(operands));
    }

    PyValue weightOp(const std::string& path, std::vector<int64_t> shape, std::string& dtype)
    {
        auto loc = builder->getUnknownLoc();
        Type elemType;
        if (dtype == "f32") {
            elemType = builder->getF32Type();
        }
        else {
            throw std::runtime_error("Unsupported dtype: " + dtype);
        }
        auto resultType = cherry::CherryTensorType::get(builder->getContext(), shape, elemType);
        auto op         = builder->create<cherry::WeightOp>(loc,
                                                    resultType,
                                                    builder->getStringAttr(path),
                                                    builder->getI64ArrayAttr(shape),
                                                    TypeAttr::get(elemType));
        return PyValue{op.getResult()};
    }

    PyValue tensorSliceOp(PyValue input, py::list starts, std::vector<int64_t> sizes, bool squeeze)
    {
        auto loc       = builder->getUnknownLoc();
        auto inputType = dyn_cast<cherry::CherryTensorType>(input.value.getType());
        if (!inputType) throw std::runtime_error("Input to tensor_slice must be a CherryTensor");

        Type               elementType = inputType.getElementType();
        std::vector<Value> startIndices;
        for (auto handle : starts) {
            try {
                startIndices.push_back(handle.cast<PyValue*>()->value);
            }
            catch (const py::cast_error&) {
                throw std::runtime_error("Start indices must be Value objects (i64)");
            }
        }
        auto sizesAttr = builder->getI64ArrayAttr(sizes);
        auto op        = builder->create<cherry::TensorSliceOp>(loc,
                                                         this->createDynamicTensorType(elementType),
                                                         input.value,
                                                         startIndices,
                                                         sizesAttr,
                                                         squeeze);
        return PyValue{op.getResult()};
    }

    PyValue tensorSetSliceOp(PyValue dest, PyValue src, py::list indices)
    {
        auto               srcType = cast<cherry::CherryTensorType>(src.value.getType());
        std::vector<Value> values;
        for (auto handle : indices) {
            values.push_back(handle.cast<PyValue*>()->value);
        }
        auto op = builder->create<cherry::TensorSetSliceOp>(
            builder->getUnknownLoc(),
            this->createDynamicTensorType(srcType.getElementType()),
            dest.value,
            src.value,
            ValueRange(values));
        return PyValue{op.getResult()};
    }

    PyValue ropeOp(PyValue input, PyValue pos)
    {
        auto inputType = cast<cherry::CherryTensorType>(input.value.getType());
        auto op        = builder->create<cherry::RopeOp>(
            builder->getUnknownLoc(),
            this->createDynamicTensorType(inputType.getElementType()),
            input.value,
            pos.value);
        return PyValue{op.getResult()};
    }

    PyValue tensorGetOp(PyValue input, py::list indices)
    {
        std::vector<Value> idxs;
        for (auto handle : indices) {
            idxs.push_back(handle.cast<PyValue*>()->value);
        }
        auto op = builder->create<cherry::TensorGetOp>(
            builder->getUnknownLoc(),
            cast<cherry::CherryTensorType>(input.value.getType()).getElementType(),
            input.value,
            ValueRange(idxs));
        return PyValue{op.getResult()};
    }

    PyValue constantOp(py::object value)
    {
        auto loc = builder->getUnknownLoc();
        if (py::isinstance<py::float_>(value)) {
            float v    = value.cast<float>();
            auto  type = builder->getF32Type();
            auto  attr = builder->getF32FloatAttr(v);
            auto  op   = builder->create<cherry::ConstantOp>(loc, type, attr);
            return PyValue{op.getResult()};
        }
        else if (py::isinstance<py::int_>(value)) {
            int  v    = value.cast<int>();
            auto type = builder->getI64Type();
            auto attr = builder->getI64IntegerAttr(v);
            auto op   = builder->create<cherry::ConstantOp>(loc, type, attr);
            return PyValue{op.getResult()};
        }
        throw std::runtime_error("Unsupported constant type: expected int or float");
    }

    PyValue createTensorOp(py::list data, std::vector<int64_t> shape)
    {
        auto               loc        = builder->getUnknownLoc();
        auto               tensorType = RankedTensorType::get(shape, builder->getF32Type());
        std::vector<float> values;
        for (auto item : data) {
            values.push_back(item.cast<float>());
        }
        ElementsAttr valueAttr = DenseElementsAttr::get(tensorType, llvm::ArrayRef(values));
        auto         op        = builder->create<cherry::CreateTensorOp>(
            loc, this->createDynamicTensorType(builder->getF32Type()), valueAttr);
        return PyValue{op.getResult()};
    }

    py::list createForLoop(int start, int stop, int step, py::list initialArgs, py::function body)
    {
        auto               loc = builder->getUnknownLoc();
        std::vector<Value> iterArgs;

        for (auto handle : initialArgs) {
            PyValue* val = handle.cast<PyValue*>();
            iterArgs.push_back(val->value);
        }

        auto startValue = builder->create<arith::ConstantOp>(
            loc, builder->getIndexType(), builder->getIndexAttr(start));
        auto stopValue = builder->create<arith::ConstantOp>(
            loc, builder->getIndexType(), builder->getIndexAttr(stop));
        auto stepValue = builder->create<arith::ConstantOp>(
            loc, builder->getIndexType(), builder->getIndexAttr(step));

        auto forOp = builder->create<scf::ForOp>(loc, startValue, stopValue, stepValue, iterArgs);
        {
            Block* bodyBlock = forOp.getBody();
            builder->setInsertionPointToStart(bodyBlock);

            py::list blockArgs;
            for (auto arg : bodyBlock->getArguments()) {
                blockArgs.append(PyValue{arg});
            }

            py::object ret = body(*blockArgs);

            std::vector<Value> yieldArgs;
            if (py::isinstance<py::list>(ret)) {
                for (auto item : ret.cast<py::list>()) {
                    yieldArgs.push_back(item.cast<PyValue*>()->value);
                }
            }
            else if (py::isinstance<py::tuple>(ret)) {
                for (auto item : ret.cast<py::tuple>()) {
                    yieldArgs.push_back(item.cast<PyValue*>()->value);
                }
            }
            else {
                if (!ret.is_none()) {
                    yieldArgs.push_back(ret.cast<PyValue*>()->value);
                }
            }
            builder->create<scf::YieldOp>(loc, yieldArgs);
        }
        builder->setInsertionPointAfter(forOp);
        py::list results;
        for (auto res : forOp.getResults()) {
            results.append(PyValue{res});
        }
        return results;
    }

    py::list createWhileLoop(py::list initialArgs, py::function cond, py::function body)
    {
        auto               loc = builder->getUnknownLoc();
        std::vector<Value> iterArgs;
        std::vector<Type>  resultTypes;
        for (auto handle : initialArgs) {
            PyValue* val = handle.cast<PyValue*>();
            iterArgs.push_back(val->value);
            resultTypes.push_back(val->value.getType());
        }

        auto whileOp = builder->create<scf::WhileOp>(loc, resultTypes, iterArgs);

        {
            Block* beforeBlock =
                builder->createBlock(&whileOp.getBefore(),
                                     {},
                                     resultTypes,
                                     std::vector<Location>(resultTypes.size(), loc));

            py::list blockArgs;
            for (auto arg : beforeBlock->getArguments()) {
                blockArgs.append(PyValue{arg});
            }
            py::object ret          = cond(*blockArgs);
            py::tuple  ret_tuple    = ret.cast<py::tuple>();
            PyValue*   condVal      = ret_tuple[0].cast<PyValue*>();
            py::list   argsToBodyPy = ret_tuple[1].cast<py::list>();

            std::vector<Value> argsToBody;
            for (auto item : argsToBodyPy) {
                argsToBody.push_back(item.cast<PyValue*>()->value);
            }
            builder->create<scf::ConditionOp>(loc, condVal->value, argsToBody);
        }
        {
            Block* afterBlock =
                builder->createBlock(&whileOp.getAfter(),
                                     {},
                                     resultTypes,
                                     std::vector<Location>(resultTypes.size(), loc));

            py::list blockArgs;
            for (auto arg : afterBlock->getArguments()) {
                blockArgs.append(PyValue{arg});
            }

            py::object ret = body(*blockArgs);

            std::vector<Value> yieldArgs;
            if (py::isinstance<py::list>(ret)) {
                for (auto item : ret.cast<py::list>()) {
                    yieldArgs.push_back(item.cast<PyValue*>()->value);
                }
            }
            else if (py::isinstance<py::tuple>(ret)) {
                for (auto item : ret.cast<py::tuple>()) {
                    yieldArgs.push_back(item.cast<PyValue*>()->value);
                }
            }
            else {
                yieldArgs.push_back(ret.cast<PyValue*>()->value);
            }

            builder->create<scf::YieldOp>(loc, yieldArgs);
        }

        builder->setInsertionPointAfter(whileOp);
        py::list results;
        for (auto res : whileOp.getResults()) {
            results.append(PyValue{res});
        }
        return results;
    }

    PyValue indexCastOp(PyValue index)
    {
        auto loc = builder->getUnknownLoc();

        auto op = builder->create<arith::IndexCastOp>(loc, builder->getI64Type(), index.value);
        return PyValue{op.getResult()};
    }

    PyValue cmpiOp(PyValue lhs, PyValue rhs, std::string pred)
    {
        auto                 loc = builder->getUnknownLoc();
        arith::CmpIPredicate predicate;
        if (pred == "eq") {
            predicate = arith::CmpIPredicate::eq;
        }
        else if (pred == "ne") {
            predicate = arith::CmpIPredicate::ne;
        }
        else if (pred == "slt") {
            predicate = arith::CmpIPredicate::slt;
        }
        else if (pred == "sle") {
            predicate = arith::CmpIPredicate::sle;
        }
        else if (pred == "sgt") {
            predicate = arith::CmpIPredicate::sgt;
        }
        else if (pred == "sge") {
            predicate = arith::CmpIPredicate::sge;
        }
        else if (pred == "ult") {
            predicate = arith::CmpIPredicate::ult;
        }
        else if (pred == "ule") {
            predicate = arith::CmpIPredicate::ule;
        }
        else if (pred == "ugt") {
            predicate = arith::CmpIPredicate::ugt;
        }
        else if (pred == "uge") {
            predicate = arith::CmpIPredicate::uge;
        }
        else {
            throw std::runtime_error("Unknown predicate: " + pred);
        }
        auto op = builder->create<arith::CmpIOp>(loc, predicate, lhs.value, rhs.value);
        return PyValue{op.getResult()};
    }

    DEFINE_SCALAR_OP(scalarAddOp, ScalarAddOp)
    DEFINE_SCALAR_OP(scalarSubOp, ScalarSubOp)
    DEFINE_SCALAR_OP(scalarMulOp, ScalarMulOp)
    DEFINE_SCALAR_OP(scalarDivOp, ScalarDivOp)

    DEFINE_TENSOR_BINARY_OP(tensorAddOp, TensorAddOp)
    DEFINE_TENSOR_BINARY_OP(tensorSubOp, TensorSubOp)
    DEFINE_TENSOR_BINARY_OP(tensorMulOp, TensorMulOp)
    DEFINE_TENSOR_BINARY_OP(tensorDivOp, TensorDivOp)
    DEFINE_TENSOR_BINARY_OP(matmulOp, MatMulOp)

    DEFINE_TENSOR_UNARY_OP(tensorNegOp, TensorNegOp)
    DEFINE_TENSOR_UNARY_OP(tensorExpOp, TensorExpOp)
    DEFINE_TENSOR_UNARY_OP(tensorReluOp, TensorReluOp)
    DEFINE_TENSOR_UNARY_OP(tensorSiluOp, TensorSiluOp)
    DEFINE_TENSOR_UNARY_OP(tensorSigmoidOp, TensorSigmoidOp)
    DEFINE_TENSOR_UNARY_OP(tensorTanhOp, TensorTanhOp)

    DEFINE_TENSOR_SCALAR_OP(tensorAddScalarOp, TensorAddScalarOp)
    DEFINE_TENSOR_SCALAR_OP(tensorSubScalarOp, TensorSubScalarOp)
    DEFINE_TENSOR_SCALAR_OP(tensorMulScalarOp, TensorMulScalarOp)
    DEFINE_TENSOR_SCALAR_OP(tensorDivScalarOp, TensorDivScalarOp)


    PyValue argmaxOp(PyValue input, int64_t dim)
    {
        return PyValue(
            builder->create<cherry::ArgMaxOp>(builder->getUnknownLoc(),
                                              this->createDynamicTensorType(builder->getI64Type()),
                                              input.value,
                                              builder->getI64IntegerAttr(dim)));
    }


    PyValue transposeOp(PyValue input, std::vector<int64_t> perm)
    {
        auto inputType = cast<cherry::CherryTensorType>(input.value.getType());
        return PyValue(builder->create<cherry::TransposeOp>(
            builder->getUnknownLoc(),
            this->createDynamicTensorType(inputType.getElementType()),
            input.value,
            builder->getI64ArrayAttr(perm)));
    }

    PyValue broadcastOp(PyValue input, py::list target_shape)
    {
        std::vector<Value> shapeVals;
        for (auto handle : target_shape) {
            shapeVals.push_back(handle.cast<PyValue*>()->value);
        }
        auto inputType = cast<cherry::CherryTensorType>(input.value.getType());
        return PyValue(builder->create<cherry::BroadcastOp>(
            builder->getUnknownLoc(),
            this->createDynamicTensorType(inputType.getElementType()),
            input.value,
            shapeVals));
    }

    PyValue maskedMatMulOp(PyValue lhs, PyValue rhs, PyValue valid_len)
    {
        auto lhsType = cast<cherry::CherryTensorType>(lhs.value.getType());
        return PyValue(builder->create<cherry::MaskedMatMulOp>(
            builder->getUnknownLoc(),
            this->createDynamicTensorType(lhsType.getElementType()),
            lhs.value,
            rhs.value,
            valid_len.value));
    }

    PyValue softmaxOp(PyValue input, int64_t axis)
    {
        auto inputType = cast<cherry::CherryTensorType>(input.value.getType());

        return PyValue(builder->create<cherry::SoftmaxOp>(
            builder->getUnknownLoc(),
            this->createDynamicTensorType(inputType.getElementType()),
            input.value,
            builder->getI64IntegerAttr(axis)));
    }


    PyValue rmsnormOp(PyValue input, PyValue scale, float eps)
    {
        auto inputType = cast<cherry::CherryTensorType>(input.value.getType());

        return PyValue(builder->create<cherry::RMSNormOp>(
            builder->getUnknownLoc(),
            this->createDynamicTensorType(inputType.getElementType()),
            input.value,
            scale.value,
            builder->getF32FloatAttr(eps)));
    }

    void printOp(PyValue input)
    {
        builder->create<cherry::PrintOp>(builder->getUnknownLoc(), input.value);
    }

    PyValue reshapeOp(PyValue input, std::vector<int64_t> shape)
    {
        auto loc = builder->getUnknownLoc();
        auto op  = builder->create<cherry::ReshapeOp>(
            loc,
            this->createDynamicTensorType(
                cast<mlir::cherry::CherryTensorType>(input.value.getType()).getElementType()),
            input.value,
            builder->getI64ArrayAttr(shape));
        return PyValue{op.getResult()};
    }

    std::string dump()
    {
        std::string              s;
        llvm::raw_string_ostream os(s);
        mlir::OpPrintingFlags    flags;
        flags.enableDebugInfo(false);
        module->print(os, flags);
        return s;
    }
    void dumpToFile(const std::string& filename)
    {
        std::error_code      ec;
        llvm::raw_fd_ostream fileStream(filename, ec);
        if (ec) {
            llvm::errs() << "Error opening file '" << filename << "': " << ec.message() << "\n";
            return;
        }
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(false);
        module->print(fileStream, flags);
    }

private:
    DialectRegistry                   registry;
    std::unique_ptr<MLIRContext>      context;
    std::unique_ptr<OpBuilder>        builder;
    mlir::OwningOpRef<mlir::ModuleOp> module;
};

PYBIND11_MODULE(core, m)
{
    m.doc() = "Cherry MLIR Python Bindings";

    py::class_<PyValue>(m, "Value");
    py::class_<PyType>(m, "Type");
    py::class_<PyCherryTensorType, PyType>(m, "CherryTensorType");

    py::class_<IrGenerator>(m, "IrGenerator")
        .def(py::init<>())
        .def("create_type", py::overload_cast<std::string>(&IrGenerator::createType))
        .def("create_type", py::overload_cast<mlir::Type>(&IrGenerator::createType))
        .def("create_tensor_type", &IrGenerator::createTensorType)
        .def("create_function", &IrGenerator::createFunction)
        .def("call", &IrGenerator::callOp)
        .def("ret", &IrGenerator::returnOp)
        .def("constant", &IrGenerator::constantOp)
        .def("runtime_call", &IrGenerator::runtimeCallOp)
        .def("load_weight", &IrGenerator::weightOp)
        .def("tensor_slice", &IrGenerator::tensorSliceOp)
        .def("dump", &IrGenerator::dump)
        .def("dump_to_file", &IrGenerator::dumpToFile)
        .def("create_tensor", &IrGenerator::createTensorOp)
        .def("create_while_loop", &IrGenerator::createWhileLoop)
        .def("create_for_loop", &IrGenerator::createForLoop)
        .def("cmpi", &IrGenerator::cmpiOp)
        .def("index_cast", &IrGenerator::indexCastOp)
        .def("scalar_add", &IrGenerator::scalarAddOp)
        .def("scalar_sub", &IrGenerator::scalarSubOp)
        .def("scalar_mul", &IrGenerator::scalarMulOp)
        .def("scalar_div", &IrGenerator::scalarDivOp)
        .def("tensor_set_slice", &IrGenerator::tensorSetSliceOp)
        .def("rope", &IrGenerator::ropeOp)
        .def("tensor_get", &IrGenerator::tensorGetOp)
        .def("reshape", &IrGenerator::reshapeOp)
        .def("transpose", &IrGenerator::transposeOp)
        .def("broadcast", &IrGenerator::broadcastOp)
        .def("add", &IrGenerator::tensorAddOp)
        .def("sub", &IrGenerator::tensorSubOp)
        .def("mul", &IrGenerator::tensorMulOp)
        .def("div", &IrGenerator::tensorDivOp)
        .def("matmul", &IrGenerator::matmulOp)
        .def("neg", &IrGenerator::tensorNegOp)
        .def("exp", &IrGenerator::tensorExpOp)
        .def("relu", &IrGenerator::tensorReluOp)
        .def("silu", &IrGenerator::tensorSiluOp)
        .def("sigmoid", &IrGenerator::tensorSigmoidOp)
        .def("tanh", &IrGenerator::tensorTanhOp)
        .def("tensor_add_scalar", &IrGenerator::tensorAddScalarOp)
        .def("tensor_sub_scalar", &IrGenerator::tensorSubScalarOp)
        .def("tensor_mul_scalar", &IrGenerator::tensorMulScalarOp)
        .def("tensor_div_scalar", &IrGenerator::tensorDivScalarOp)
        .def("argmax", &IrGenerator::argmaxOp)
        .def("masked_matmul", &IrGenerator::maskedMatMulOp)
        .def("softmax", &IrGenerator::softmaxOp)
        .def("rmsnorm", &IrGenerator::rmsnormOp)
        .def("print", &IrGenerator::printOp);
    ;
}
