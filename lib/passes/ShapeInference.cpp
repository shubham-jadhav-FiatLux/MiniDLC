#include "dlc/passes/ShapeInference.h"
#include "dlc/ir/Ops.h"
#include <algorithm>

namespace dlc {

bool ShapeInferencePass::broadcastShapes(const Shape& a, const Shape& b, Shape& outShape) {
    size_t rankA = a.size();
    size_t rankB = b.size();
    size_t maxRank = std::max(rankA, rankB);
    outShape.resize(maxRank);

    for (size_t i = 0; i < maxRank; ++i) {
        int64_t dimA = (i < rankA) ? a[rankA - 1 - i] : 1;
        int64_t dimB = (i < rankB) ? b[rankB - 1 - i] : 1;

        if (dimA == dimB) {
            outShape[maxRank - 1 - i] = dimA;
        } else if (dimA == 1) {
            outShape[maxRank - 1 - i] = dimB;
        } else if (dimB == 1) {
            outShape[maxRank - 1 - i] = dimA;
        } else {
            return false; // Incompatible shapes
        }
    }
    return true;
}

bool ShapeInferencePass::runOnFunction(Function& func) {
    bool changed = false;
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    for (auto& op : entry->getOperations()) {
        const std::string& name = op->getOpName();
        if (op->getNumResults() == 0) continue;

        ValuePtr res = op->getResult(0);
        TypePtr expected = nullptr;

        if (name == "add" || name == "sub" || name == "mul" || name == "div") {
            if (op->getNumOperands() >= 2 && op->getOperand(0) && op->getOperand(1)) {
                auto op0 = op->getOperand(0)->getType();
                auto op1 = op->getOperand(1)->getType();
                if (op0 && op1 && op0->isTensor() && op1->isTensor()) {
                    auto t0 = std::static_pointer_cast<TensorType>(op0);
                    auto t1 = std::static_pointer_cast<TensorType>(op1);
                    Shape outShape;
                    if (broadcastShapes(t0->getShape(), t1->getShape(), outShape)) {
                        expected = std::make_shared<TensorType>(t0->getElementType(), outShape);
                    }
                }
            }
        } else if (name == "matmul") {
            if (op->getNumOperands() >= 2 && op->getOperand(0) && op->getOperand(1)) {
                auto op0 = op->getOperand(0)->getType();
                auto op1 = op->getOperand(1)->getType();
                if (op0 && op1 && op0->isTensor() && op1->isTensor()) {
                    auto t0 = std::static_pointer_cast<TensorType>(op0);
                    auto t1 = std::static_pointer_cast<TensorType>(op1);
                    const auto& s0 = t0->getShape();
                    const auto& s1 = t1->getShape();
                    if (s0.size() == 2 && s1.size() == 2 && s0[1] == s1[0]) {
                        expected = std::make_shared<TensorType>(t0->getElementType(), Shape{s0[0], s1[1]});
                    } else if (s0.size() == 3 && s1.size() == 3 && s0[0] == s1[0] && s0[2] == s1[1]) {
                        expected = std::make_shared<TensorType>(t0->getElementType(), Shape{s0[0], s0[1], s1[2]});
                    } else if (s0.size() == 4 && s1.size() == 4 && s0[0] == s1[0] && s0[1] == s1[1] && s0[3] == s1[2]) {
                        expected = std::make_shared<TensorType>(t0->getElementType(), Shape{s0[0], s0[1], s0[2], s1[3]});
                    }
                }
            }
        } else if (name == "relu" || name == "gelu" || name == "exp" || name == "log" ||
                   name == "softmax" || name == "log_softmax") {
            if (op->getNumOperands() >= 1 && op->getOperand(0)) {
                expected = op->getOperand(0)->getType();
            }
        } else if (name == "cross_entropy") {
            expected = std::make_shared<TensorType>(DataType::Float32, Shape{});
        } else if (name == "layernorm") {
            if (op->getNumOperands() >= 1 && op->getOperand(0)) {
                expected = op->getOperand(0)->getType();
            }
        } else if (name == "sum" || name == "mean") {
            if (op->getNumOperands() >= 1 && op->getOperand(0) && op->getOperand(0)->getType()->isTensor()) {
                auto t0 = std::static_pointer_cast<TensorType>(op->getOperand(0)->getType());
                int64_t axis = op->hasAttribute("axis") ? std::get<int64_t>(op->getAttribute("axis")) : -1;
                bool keepdims = op->hasAttribute("keepdims") && (std::get<int64_t>(op->getAttribute("keepdims")) != 0);
                Shape s = t0->getShape();
                if (axis < 0) axis += static_cast<int64_t>(s.size());
                if (axis >= 0 && axis < static_cast<int64_t>(s.size())) {
                    if (keepdims) {
                        s[axis] = 1;
                    } else {
                        s.erase(s.begin() + axis);
                    }
                    expected = std::make_shared<TensorType>(t0->getElementType(), s);
                }
            }
        } else if (name == "transpose") {
            if (op->getNumOperands() >= 1 && op->getOperand(0) && op->getOperand(0)->getType()->isTensor()) {
                auto t0 = std::static_pointer_cast<TensorType>(op->getOperand(0)->getType());
                if (op->hasAttribute("perm")) {
                    auto perm = std::get<std::vector<int64_t>>(op->getAttribute("perm"));
                    const auto& s = t0->getShape();
                    if (perm.size() == s.size()) {
                        Shape newShape(perm.size());
                        for (size_t i = 0; i < perm.size(); ++i) {
                            newShape[i] = s[perm[i]];
                        }
                        expected = std::make_shared<TensorType>(t0->getElementType(), newShape);
                    }
                }
            }
        } else if (name == "reshape") {
            if (op->hasAttribute("shape")) {
                auto newShape = std::get<Shape>(op->getAttribute("shape"));
                DataType dt = DataType::Float32;
                if (op->getNumOperands() >= 1 && op->getOperand(0) && op->getOperand(0)->getType()->isTensor()) {
                    dt = std::static_pointer_cast<TensorType>(op->getOperand(0)->getType())->getElementType();
                }
                expected = std::make_shared<TensorType>(dt, newShape);
            }
        } else if (name == "embedding") {
            if (op->getNumOperands() >= 2 && op->getOperand(0) && op->getOperand(1)) {
                auto tTable = std::static_pointer_cast<TensorType>(op->getOperand(0)->getType());
                auto tIdx = std::static_pointer_cast<TensorType>(op->getOperand(1)->getType());
                if (tTable && tIdx && tTable->getShape().size() >= 2) {
                    Shape outShape = tIdx->getShape();
                    outShape.push_back(tTable->getShape()[1]);
                    expected = std::make_shared<TensorType>(tTable->getElementType(), outShape);
                }
            }
        } else if (name == "onehot") {
            if (op->getNumOperands() >= 1 && op->getOperand(0) && op->hasAttribute("num_classes")) {
                auto tIdx = std::static_pointer_cast<TensorType>(op->getOperand(0)->getType());
                int64_t numClasses = std::get<int64_t>(op->getAttribute("num_classes"));
                if (tIdx) {
                    Shape outShape = tIdx->getShape();
                    outShape.push_back(numClasses);
                    expected = std::make_shared<TensorType>(DataType::Float32, outShape);
                }
            }
        }

        if (expected && (!res->getType() || res->getType()->toString() != expected->toString())) {
            res->setType(expected);
            changed = true;
        }
    }

    return changed;
}

} // namespace dlc
