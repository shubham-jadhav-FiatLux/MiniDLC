#include "dlc/ir/Builder.h"
#include "dlc/ir/OpRegistry.h"

namespace dlc {

ValuePtr IRBuilder::createValue(TypePtr type, std::string name) {
    return std::make_shared<Value>(nextValueId_++, std::move(type), nullptr, std::move(name));
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

ValuePtr IRBuilder::insert(OperationPtr op, const TypePtr& explicitType) {
    finalizeOp(*op, /*inferResultType=*/explicitType == nullptr);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getNumResults() > 0 ? op->getResult(0) : nullptr;
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

void IRBuilder::createReturn(const std::vector<ValuePtr>& values) {
    auto op = std::make_shared<ReturnOp>(values);
    if (insertBlock_) insertBlock_->addOperation(op);
}

} // namespace dlc
