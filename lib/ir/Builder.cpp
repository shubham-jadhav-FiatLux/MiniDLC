#include "dlc/ir/Builder.h"
#include <cassert>

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

ValuePtr IRBuilder::createMatmul(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    if (!resultType && lhs && rhs && lhs->getType()->isTensor() && rhs->getType()->isTensor()) {
        auto t1 = std::static_pointer_cast<TensorType>(lhs->getType());
        auto t2 = std::static_pointer_cast<TensorType>(rhs->getType());
        const auto& s1 = t1->getShape();
        const auto& s2 = t2->getShape();
        Shape resShape;
        if (s1.size() == 2 && s2.size() == 2) {
            resShape = {s1[0], s2[1]};
        } else if (s1.size() == 3 && s2.size() == 3) {
            resShape = {s1[0], s1[1], s2[2]};
        }
        resultType = std::make_shared<TensorType>(t1->getElementType(), resShape);
    }
    auto op = std::make_shared<MatmulOp>(lhs, rhs, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

static TypePtr inferBinaryBroadcastType(ValuePtr lhs, ValuePtr rhs) {
    if (lhs && rhs && lhs->getType()->isTensor() && rhs->getType()->isTensor()) {
        auto t1 = std::static_pointer_cast<TensorType>(lhs->getType());
        auto t2 = std::static_pointer_cast<TensorType>(rhs->getType());
        Shape outShape;
        if (computeBroadcastShape(t1->getShape(), t2->getShape(), outShape)) {
            return std::make_shared<TensorType>(t1->getElementType(), outShape);
        }
    }
    if (lhs) return lhs->getType();
    if (rhs) return rhs->getType();
    return nullptr;
}

ValuePtr IRBuilder::createAdd(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    if (!resultType) resultType = inferBinaryBroadcastType(lhs, rhs);
    auto op = std::make_shared<AddOp>(lhs, rhs, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createSub(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    if (!resultType) resultType = inferBinaryBroadcastType(lhs, rhs);
    auto op = std::make_shared<SubOp>(lhs, rhs, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createMul(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    if (!resultType) resultType = inferBinaryBroadcastType(lhs, rhs);
    auto op = std::make_shared<MulOp>(lhs, rhs, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createDiv(ValuePtr lhs, ValuePtr rhs, TypePtr resultType) {
    if (!resultType) resultType = inferBinaryBroadcastType(lhs, rhs);
    auto op = std::make_shared<DivOp>(lhs, rhs, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createRelu(ValuePtr operand, TypePtr resultType) {
    if (!resultType && operand) resultType = operand->getType();
    auto op = std::make_shared<ReluOp>(operand, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createGelu(ValuePtr operand, TypePtr resultType) {
    if (!resultType && operand) resultType = operand->getType();
    auto op = std::make_shared<GeluOp>(operand, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createExp(ValuePtr operand, TypePtr resultType) {
    if (!resultType && operand) resultType = operand->getType();
    auto op = std::make_shared<ExpOp>(operand, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createLog(ValuePtr operand, TypePtr resultType) {
    if (!resultType && operand) resultType = operand->getType();
    auto op = std::make_shared<LogOp>(operand, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createSoftmax(ValuePtr operand, int64_t axis, TypePtr resultType) {
    if (!resultType && operand) resultType = operand->getType();
    auto op = std::make_shared<SoftmaxOp>(operand, axis, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createLogSoftmax(ValuePtr operand, int64_t axis, TypePtr resultType) {
    if (!resultType && operand) resultType = operand->getType();
    auto op = std::make_shared<LogSoftmaxOp>(operand, axis, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createCrossEntropy(ValuePtr logits, ValuePtr targets, TypePtr resultType) {
    if (!resultType) {
        resultType = std::make_shared<TensorType>(DataType::Float32, Shape{});
    }
    auto op = std::make_shared<CrossEntropyOp>(logits, targets, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createLayerNorm(ValuePtr x, ValuePtr gamma, ValuePtr beta, double eps, TypePtr resultType) {
    if (!resultType && x) resultType = x->getType();
    auto op = std::make_shared<LayerNormOp>(x, gamma, beta, eps, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createSum(ValuePtr operand, int64_t axis, bool keepdims, TypePtr resultType) {
    if (!resultType && operand && operand->getType()->isTensor()) {
        auto t = std::static_pointer_cast<TensorType>(operand->getType());
        Shape s = t->getShape();
        if (axis < 0) axis += s.size();
        if (axis >= 0 && axis < static_cast<int64_t>(s.size())) {
            if (keepdims) {
                s[axis] = 1;
            } else {
                s.erase(s.begin() + axis);
            }
        }
        resultType = std::make_shared<TensorType>(t->getElementType(), s);
    }
    auto op = std::make_shared<SumOp>(operand, axis, keepdims, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createMean(ValuePtr operand, int64_t axis, bool keepdims, TypePtr resultType) {
    if (!resultType && operand && operand->getType()->isTensor()) {
        auto t = std::static_pointer_cast<TensorType>(operand->getType());
        Shape s = t->getShape();
        if (axis < 0) axis += s.size();
        if (axis >= 0 && axis < static_cast<int64_t>(s.size())) {
            if (keepdims) {
                s[axis] = 1;
            } else {
                s.erase(s.begin() + axis);
            }
        }
        resultType = std::make_shared<TensorType>(t->getElementType(), s);
    }
    auto op = std::make_shared<MeanOp>(operand, axis, keepdims, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createTranspose(ValuePtr operand, std::vector<int64_t> perm, TypePtr resultType) {
    if (!resultType && operand && operand->getType()->isTensor()) {
        auto t = std::static_pointer_cast<TensorType>(operand->getType());
        const auto& s = t->getShape();
        Shape newShape(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            newShape[i] = s[perm[i]];
        }
        resultType = std::make_shared<TensorType>(t->getElementType(), newShape);
    }
    auto op = std::make_shared<TransposeOp>(operand, perm, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createReshape(ValuePtr operand, Shape newShape, TypePtr resultType) {
    if (!resultType && operand && operand->getType()->isTensor()) {
        auto t = std::static_pointer_cast<TensorType>(operand->getType());
        resultType = std::make_shared<TensorType>(t->getElementType(), newShape);
    }
    auto op = std::make_shared<ReshapeOp>(operand, newShape, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createEmbedding(ValuePtr table, ValuePtr indices, TypePtr resultType) {
    if (!resultType && table && indices && table->getType()->isTensor() && indices->getType()->isTensor()) {
        auto tTable = std::static_pointer_cast<TensorType>(table->getType());
        auto tIdx = std::static_pointer_cast<TensorType>(indices->getType());
        Shape s = tIdx->getShape();
        if (tTable->getShape().size() > 1) {
            s.push_back(tTable->getShape()[1]);
        }
        resultType = std::make_shared<TensorType>(tTable->getElementType(), s);
    }
    auto op = std::make_shared<EmbeddingOp>(table, indices, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createConstant(double value, TypePtr resultType) {
    auto op = std::make_shared<ConstantOp>(value, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

ValuePtr IRBuilder::createOneHot(ValuePtr indices, int64_t numClasses, TypePtr resultType) {
    if (!resultType && indices && indices->getType()->isTensor()) {
        auto tIdx = std::static_pointer_cast<TensorType>(indices->getType());
        Shape s = tIdx->getShape();
        s.push_back(numClasses);
        resultType = std::make_shared<TensorType>(DataType::Float32, s);
    }
    auto op = std::make_shared<OneHotOp>(indices, numClasses, resultType);
    if (insertBlock_) insertBlock_->addOperation(op);
    return op->getResult(0);
}

void IRBuilder::createReturn(const std::vector<ValuePtr>& values) {
    auto op = std::make_shared<ReturnOp>(values);
    if (insertBlock_) insertBlock_->addOperation(op);
}

} // namespace dlc
