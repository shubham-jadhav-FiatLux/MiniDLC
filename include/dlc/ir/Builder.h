#pragma once

#include "dlc/ir/Module.h"
#include "dlc/ir/Ops.h"
#include <map>
#include <memory>
#include <optional>

namespace dlc {

// Creates ops at the end of a block. Result types come from the op registry unless one is passed explicitly.
class IRBuilder {
public:
    IRBuilder() = default;
    explicit IRBuilder(Block* block) : insertBlock_(block) {}

    void setInsertBlock(Block* block) { insertBlock_ = block; }
    Block* getInsertBlock() const { return insertBlock_; }

    ValuePtr createValue(TypePtr type, std::string name = "");

    FunctionPtr createFunction(const std::string& name,
                               const std::vector<TypePtr>& argTypes,
                               const std::vector<TypePtr>& resultTypes);

    // Any registered op by name.
    ValuePtr createOp(const std::string& opName, std::vector<ValuePtr> operands,
                      const std::map<std::string, AttributeValue>& attributes = {}, TypePtr resultType = nullptr);

    // High level op builders
    ValuePtr createMatmul(ValuePtr lhs, ValuePtr rhs, TypePtr resultType = nullptr);
    ValuePtr createAdd(ValuePtr lhs, ValuePtr rhs, TypePtr resultType = nullptr);
    ValuePtr createSub(ValuePtr lhs, ValuePtr rhs, TypePtr resultType = nullptr);
    ValuePtr createMul(ValuePtr lhs, ValuePtr rhs, TypePtr resultType = nullptr);
    ValuePtr createDiv(ValuePtr lhs, ValuePtr rhs, TypePtr resultType = nullptr);

    ValuePtr createRelu(ValuePtr operand, TypePtr resultType = nullptr);
    ValuePtr createGelu(ValuePtr operand, TypePtr resultType = nullptr);
    ValuePtr createExp(ValuePtr operand, TypePtr resultType = nullptr);
    ValuePtr createLog(ValuePtr operand, TypePtr resultType = nullptr);

    ValuePtr createSoftmax(ValuePtr operand, int64_t axis = -1, TypePtr resultType = nullptr);
    ValuePtr createLogSoftmax(ValuePtr operand, int64_t axis = -1, TypePtr resultType = nullptr);
    ValuePtr createCrossEntropy(ValuePtr logits, ValuePtr targets, TypePtr resultType = nullptr);
    ValuePtr createLayerNorm(ValuePtr x, ValuePtr gamma, ValuePtr beta, double eps = 1e-5, TypePtr resultType = nullptr);

    // Without `axes` the reduction covers every axis.
    ValuePtr createSum(ValuePtr operand, std::optional<std::vector<int64_t>> axes = std::nullopt,
                       bool keepdims = false, TypePtr resultType = nullptr);
    ValuePtr createMean(ValuePtr operand, std::optional<std::vector<int64_t>> axes = std::nullopt,
                        bool keepdims = false, TypePtr resultType = nullptr);

    // An empty `perm` reverses the axes.
    ValuePtr createTranspose(ValuePtr operand, std::vector<int64_t> perm, TypePtr resultType = nullptr);
    ValuePtr createReshape(ValuePtr operand, Shape newShape, TypePtr resultType = nullptr);
    ValuePtr createEmbedding(ValuePtr table, ValuePtr indices, TypePtr resultType = nullptr);
    ValuePtr createConstant(double value, TypePtr resultType);
    ValuePtr createOneHot(ValuePtr indices, int64_t numClasses, TypePtr resultType = nullptr);

    void createReturn(const std::vector<ValuePtr>& values);

private:
    // Fills default attributes, infers the result type unless `explicitType` is given, and appends the op.
    ValuePtr insert(OperationPtr op, const TypePtr& explicitType);

    Block* insertBlock_{nullptr};
    uint32_t nextValueId_{0};
};

} // namespace dlc
