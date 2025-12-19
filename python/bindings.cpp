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

struct PyValue
{
    Value value;
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
        // builder->setInsertionPointToEnd(module.getBody());
    }

    PyType create_type(std::string name)
    {
        if (name == "f32") return PyType(builder->getF32Type());
        if (name == "i32") return PyType(builder->getI32Type());
        if (name == "i64") return PyType(builder->getI64Type());
        if (name == "index") return PyType(builder->getIndexType());
        throw std::runtime_error("Unknown scalar type: " + name);
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

    void createFunction(std::string name, std::vector<PyType*> args = {},
                        std::vector<PyType*> rets = {}, bool isPrivate = false)
    {
        builder->setInsertionPointToEnd(module.getBody());
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


    void returnOp(py::args args)
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

        builder->create<cherry::ReturnOp>(loc, operands);
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
            auto type = builder->getI32Type();
            auto attr = builder->getI32IntegerAttr(v);
            auto op   = builder->create<cherry::ConstantOp>(loc, type, attr);
            return PyValue{op.getResult()};
        }
        throw std::runtime_error("Unsupported constant type: expected int or float");
    }

    PyValue createTensorOp(py::list data, std::vector<int64_t> shape)
    {
        auto loc = builder->getUnknownLoc();
        auto resultType =
            cherry::CherryTensorType::get(builder->getContext(), shape, builder->getF32Type());
        auto               tensorType = RankedTensorType::get(shape, builder->getF32Type());
        std::vector<float> values;
        for (auto item : data) {
            values.push_back(item.cast<float>());
        }
        ElementsAttr valueAttr = DenseElementsAttr::get(tensorType, llvm::ArrayRef(values));
        auto         op = builder->create<cherry::CreateTensorOp>(loc, resultType, valueAttr);
        return PyValue{op.getResult()};
    }

    py::list createLoop(py::list initialArgs, py::function cond, py::function body)
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

    PyValue scalarAddOp(PyValue lhs, PyValue rhs)
    {
        auto loc        = builder->getUnknownLoc();
        Type resultType = lhs.value.getType();
        auto op = builder->create<cherry::ScalarAddOp>(loc, resultType, lhs.value, rhs.value);
        return PyValue{op.getResult()};
    }

    PyValue scalarSubOp(PyValue lhs, PyValue rhs)
    {
        auto loc        = builder->getUnknownLoc();
        Type resultType = lhs.value.getType();
        auto op = builder->create<cherry::ScalarSubOp>(loc, resultType, lhs.value, rhs.value);
        return PyValue{op.getResult()};
    }

    PyValue scalarMulOp(PyValue lhs, PyValue rhs)
    {
        auto loc        = builder->getUnknownLoc();
        Type resultType = lhs.value.getType();
        auto op = builder->create<cherry::ScalarMulOp>(loc, resultType, lhs.value, rhs.value);
        return PyValue{op.getResult()};
    }

    PyValue scalarDivOp(PyValue lhs, PyValue rhs)
    {
        auto loc        = builder->getUnknownLoc();
        Type resultType = lhs.value.getType();
        auto op = builder->create<cherry::ScalarDivOp>(loc, resultType, lhs.value, rhs.value);
        return PyValue{op.getResult()};
    }

    std::string dump()
    {
        std::string              s;
        llvm::raw_string_ostream os(s);
        module->print(os);
        return s;
    }

private:
    DialectRegistry              registry;
    std::unique_ptr<MLIRContext> context;
    std::unique_ptr<OpBuilder>   builder;
    ModuleOp                     module;
};

PYBIND11_MODULE(core, m)
{
    m.doc() = "Cherry MLIR Python Bindings";

    py::class_<PyValue>(m, "Value");
    py::class_<PyType>(m, "Type");
    py::class_<PyCherryTensorType, PyType>(m, "CherryTensorType");

    py::class_<IrGenerator>(m, "IrGenerator")
        .def(py::init<>())
        .def("create_type", &IrGenerator::create_type)
        .def("create_tensor_type", &IrGenerator::createTensorType)
        .def("create_function", &IrGenerator::createFunction)
        .def("ret", &IrGenerator::returnOp)
        .def("constant", &IrGenerator::constantOp)
        .def("runtime_call", &IrGenerator::runtimeCallOp)
        .def("load_weight", &IrGenerator::weightOp)
        .def("dump", &IrGenerator::dump)
        .def("create_tensor", &IrGenerator::createTensorOp)
        .def("create_loop", &IrGenerator::createLoop)
        .def("cmpi", &IrGenerator::cmpiOp)
        .def("scalar_add", &IrGenerator::scalarAddOp)
        .def("scalar_sub", &IrGenerator::scalarSubOp)
        .def("scalar_mul", &IrGenerator::scalarMulOp)
        .def("scalar_div", &IrGenerator::scalarDivOp);
}
