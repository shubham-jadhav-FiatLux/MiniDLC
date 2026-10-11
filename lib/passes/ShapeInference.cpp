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
        if (name == "add" || name == "sub" || name == "mul" || name == "div") {
            if (op->getNumOperands() == 2 && op->getNumResults() == 1) {
                auto op0 = op->getOperand(0);
                auto op1 = op->getOperand(1);
                if (op0 && op1 && op0->getType()->isTensor() && op1->getType()->isTensor()) {
                    auto t0 = std::static_pointer_cast<TensorType>(op0->getType());
                    auto t1 = std::static_pointer_cast<TensorType>(op1->getType());
                    Shape outShape;
                    if (broadcastShapes(t0->getShape(), t1->getShape(), outShape)) {
                        auto curRes = op->getResult(0);
                        auto expected = std::make_shared<TensorType>(t0->getElementType(), outShape);
                        if (!curRes->getType() || curRes->getType()->toString() != expected->toString()) {
                            curRes = std::make_shared<Value>(curRes->getId(), expected, op.get());
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    return changed;
}

} // namespace dlc
