#include "dlc/ir/Builder.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/OpRegistry.h"

namespace dlc {

ValuePtr IRBuilder::createValue(TypePtr type, std::string name) {
    return std::make_shared<Value>(Value::nextId(), std::move(type), nullptr, std::move(name));
}

FunctionPtr IRBuilder::createFunction(const std::string& name,
                                     const std::vector<TypePtr>& argTypes,
                                     const std::vector<TypePtr>& resultTypes) {
    std::vector<ValuePtr> args;
    args.reserve(argTypes.size());
    for (size_t i = 0; i < argTypes.size(); ++i) {
        args.push_back(createValue(argTypes[i], "arg" + std::to_string(i)));
    }
    auto func = std::make_shared<Function>(name, std::move(args), resultTypes);
    insertBlock_ = func->getEntryBlock();
    return func;
}

void IRBuilder::place(OperationPtr op) {
    if (!insertBlock_) return;
    if (!insertBefore_) {
        insertBlock_->addOperation(std::move(op));
        return;
    }
    const auto& ops = insertBlock_->getOperations();
    for (size_t i = 0; i < ops.size(); ++i) {
        if (ops[i].get() == insertBefore_) {
            insertBlock_->insertOperation(i, std::move(op));
            return;
        }
    }
    throw IRError("IRBuilder: the insertion point is not in its block");
}

ValuePtr IRBuilder::insert(OperationPtr op, const TypePtr& explicitType) {
    finalizeOp(*op, /*inferResultType=*/explicitType == nullptr);
    ValuePtr result = op->getNumResults() > 0 ? op->getResult(0) : nullptr;
    place(std::move(op));
    return result;
}

ValuePtr IRBuilder::createOp(const std::string& opName, std::vector<ValuePtr> operands,
                             const std::map<std::string, AttributeValue>& attributes, TypePtr resultType) {
    auto op = std::make_shared<Operation>(opName, std::move(operands), std::vector<TypePtr>{resultType});
    for (const auto& [key, value] : attributes) op->setAttribute(key, value);
    return insert(op, resultType);
}

ValuePtr IRBuilder::createMatmul(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    return insert(std::make_shared<MatmulOp>(lhs, rhs, resultType), resultType);
}

ValuePtr IRBuilder::createAdd(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    return insert(std::make_shared<AddOp>(lhs, rhs, resultType), resultType);
}

ValuePtr IRBuilder::createSub(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    return insert(std::make_shared<SubOp>(lhs, rhs, resultType), resultType);
}

ValuePtr IRBuilder::createMul(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    return insert(std::make_shared<MulOp>(lhs, rhs, resultType), resultType);
}

ValuePtr IRBuilder::createDiv(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    return insert(std::make_shared<DivOp>(lhs, rhs, resultType), resultType);
}

ValuePtr IRBuilder::createRelu(ValuePtr operand, TypePtr resultType) {
    return insert(std::make_shared<ReluOp>(operand, resultType), resultType);
}

ValuePtr IRBuilder::createGelu(ValuePtr operand, TypePtr resultType) {
    return insert(std::make_shared<GeluOp>(operand, resultType), resultType);
}

ValuePtr IRBuilder::createExp(ValuePtr operand, TypePtr resultType) {
    return insert(std::make_shared<ExpOp>(operand, resultType), resultType);
}

ValuePtr IRBuilder::createLog(ValuePtr operand, TypePtr resultType) {
    return insert(std::make_shared<LogOp>(operand, resultType), resultType);
}

ValuePtr IRBuilder::createSoftmax(ValuePtr operand, int64_t axis, TypePtr resultType) {
    return insert(std::make_shared<SoftmaxOp>(operand, axis, resultType), resultType);
}

ValuePtr IRBuilder::createLogSoftmax(ValuePtr operand, int64_t axis, TypePtr resultType) {
    return insert(std::make_shared<LogSoftmaxOp>(operand, axis, resultType), resultType);
}

ValuePtr IRBuilder::createCrossEntropy(ValuePtr logits, ValuePtr targets, TypePtr resultType) {
    return insert(std::make_shared<CrossEntropyOp>(logits, targets, resultType), resultType);
}

ValuePtr IRBuilder::createLayerNorm(ValuePtr x, ValuePtr gamma, ValuePtr beta, double eps, TypePtr resultType) {
    return insert(std::make_shared<LayerNormOp>(x, gamma, beta, eps, resultType), resultType);
}

ValuePtr IRBuilder::createSum(ValuePtr operand, std::optional<std::vector<int64_t>> axes, bool keepdims,
                              TypePtr resultType) {
    return insert(std::make_shared<SumOp>(operand, std::move(axes), keepdims, resultType), resultType);
}

ValuePtr IRBuilder::createMean(ValuePtr operand, std::optional<std::vector<int64_t>> axes, bool keepdims,
                               TypePtr resultType) {
    return insert(std::make_shared<MeanOp>(operand, std::move(axes), keepdims, resultType), resultType);
}

ValuePtr IRBuilder::createTranspose(ValuePtr operand, std::vector<int64_t> perm, TypePtr resultType) {
    return insert(std::make_shared<TransposeOp>(operand, std::move(perm), resultType), resultType);
}

ValuePtr IRBuilder::createReshape(ValuePtr operand, Shape newShape, TypePtr resultType) {
    return insert(std::make_shared<ReshapeOp>(operand, std::move(newShape), resultType), resultType);
}

ValuePtr IRBuilder::createEmbedding(ValuePtr table, ValuePtr indices, TypePtr resultType) {
    return insert(std::make_shared<EmbeddingOp>(table, indices, resultType), resultType);
}

ValuePtr IRBuilder::createConstant(double value, TypePtr resultType) {
    return insert(std::make_shared<ConstantOp>(value, resultType), resultType);
}

ValuePtr IRBuilder::createOneHot(ValuePtr indices, int64_t numClasses, TypePtr resultType) {
    return insert(std::make_shared<OneHotOp>(indices, numClasses, resultType), resultType);
}

ValuePtr IRBuilder::createMatmul(ValuePtr lhs, ValuePtr rhs, bool transposeA, bool transposeB) {
    std::map<std::string, AttributeValue> attrs;
    if (transposeA) attrs["transpose_a"] = int64_t{1};
    if (transposeB) attrs["transpose_b"] = int64_t{1};
    return createOp("matmul", {std::move(lhs), std::move(rhs)}, attrs);
}

ValuePtr IRBuilder::createUnary(const std::string& opName, ValuePtr operand) {
    return createOp(opName, {std::move(operand)});
}

ValuePtr IRBuilder::createBinary(const std::string& opName, ValuePtr lhs, ValuePtr rhs) {
    return createOp(opName, {std::move(lhs), std::move(rhs)});
}

ValuePtr IRBuilder::createSelect(ValuePtr cond, ValuePtr ifTrue, ValuePtr ifFalse) {
    return createOp("select", {std::move(cond), std::move(ifTrue), std::move(ifFalse)});
}

ValuePtr IRBuilder::createBroadcastTo(ValuePtr operand, Shape shape) {
    return createOp("broadcast_to", {std::move(operand)}, {{"shape", std::move(shape)}});
}

ValuePtr IRBuilder::createReduce(const std::string& opName, ValuePtr operand, std::vector<int64_t> axes,
                                 bool keepdims) {
    return createOp(opName, {std::move(operand)},
                    {{"axes", std::move(axes)}, {"keepdims", static_cast<int64_t>(keepdims ? 1 : 0)}});
}

ValuePtr IRBuilder::createScatterAdd(ValuePtr indices, ValuePtr updates, int64_t numRows) {
    return createOp("scatter_add", {std::move(indices), std::move(updates)}, {{"num_rows", numRows}});
}

ValuePtr IRBuilder::createScalar(double value) {
    return createConstant(value, std::make_shared<TensorType>(DataType::Float32, Shape{}));
}

void IRBuilder::createReturn(const std::vector<ValuePtr>& values) {
    place(std::make_shared<ReturnOp>(values));
}

} // namespace dlc
