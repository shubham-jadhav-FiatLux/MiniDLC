#include "dlc/passes/Autodiff.h"
#include "dlc/ir/Builder.h"
#include <unordered_map>
#include <vector>

namespace dlc {

static Shape getShape(ValuePtr val) {
    if (val && val->getType()->isTensor()) {
        return std::static_pointer_cast<TensorType>(val->getType())->getShape();
    }
    return {};
}

bool AutodiffPass::runOnFunction(Function& func) {
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    // Identify return op (assumed scalar loss for autodiff)
    Operation* returnOp = nullptr;
    for (const auto& opPtr : entry->getOperations()) {
        if (opPtr->getOpName() == "return") {
            returnOp = opPtr.get();
            break;
        }
    }
    if (!returnOp || returnOp->getNumOperands() == 0) {
        return false;
    }

    ValuePtr lossVal = returnOp->getOperand(0);
    IRBuilder builder(entry);

    // Seed adjoint dL/dLoss = 1.0
    auto scalarF32 = std::make_shared<TensorType>(DataType::Float32, Shape{});
    ValuePtr oneConst = builder.createConstant(1.0, scalarF32);

    std::unordered_map<Value*, ValuePtr> adjoints;
    adjoints[lossVal.get()] = oneConst;

    auto accumulateAdjoint = [&](ValuePtr target, ValuePtr grad) {
        if (!target || !grad) return;
        auto it = adjoints.find(target.get());
        if (it == adjoints.end()) {
            adjoints[target.get()] = grad;
        } else {
            adjoints[target.get()] = builder.createAdd(it->second, grad);
        }
    };

    auto reduceBroadcast = [&](ValuePtr dY, const Shape& targetShape) -> ValuePtr {
        if (!dY || !dY->getType()->isTensor()) return dY;
        auto tY = std::static_pointer_cast<TensorType>(dY->getType());
        const auto& yShape = tY->getShape();
        if (yShape == targetShape) return dY;

        ValuePtr curr = dY;
        if (targetShape.empty()) {
            return builder.createSum(curr, -1, false);
        }

        int64_t extraDims = static_cast<int64_t>(yShape.size()) - static_cast<int64_t>(targetShape.size());
        for (int64_t i = 0; i < extraDims; ++i) {
            curr = builder.createSum(curr, 0, false);
        }

        for (size_t i = 0; i < targetShape.size(); ++i) {
            if (targetShape[i] == 1 && i < getShape(curr).size() && getShape(curr)[i] > 1) {
                curr = builder.createSum(curr, static_cast<int64_t>(i), true);
            }
        }
        return curr;
    };

    // Snapshot forward operations before adding backward ops
    std::vector<OperationPtr> forwardOps = entry->getOperations();

    // Remove the forward return op temporarily to append backward pass
    entry->removeOperation(returnOp);

    // Reverse topological traversal
    for (auto it = forwardOps.rbegin(); it != forwardOps.rend(); ++it) {
        Operation* op = it->get();
        if (op->getOpName() == "return") continue;
        if (op->getNumResults() == 0) continue;

        ValuePtr outVal = op->getResult(0);
        if (adjoints.find(outVal.get()) == adjoints.end()) {
            continue;
        }

        ValuePtr dOut = adjoints[outVal.get()];
        const std::string& opName = op->getOpName();

        if (opName == "add") {
            ValuePtr aVal = op->getOperand(0);
            ValuePtr bVal = op->getOperand(1);
            accumulateAdjoint(aVal, reduceBroadcast(dOut, getShape(aVal)));
            accumulateAdjoint(bVal, reduceBroadcast(dOut, getShape(bVal)));
        } else if (opName == "sub") {
            ValuePtr aVal = op->getOperand(0);
            ValuePtr bVal = op->getOperand(1);
            accumulateAdjoint(aVal, reduceBroadcast(dOut, getShape(aVal)));
            ValuePtr negOne = builder.createConstant(-1.0, scalarF32);
            ValuePtr negDOut = builder.createMul(dOut, negOne);
            accumulateAdjoint(bVal, reduceBroadcast(negDOut, getShape(bVal)));
        } else if (opName == "mul") {
            ValuePtr aVal = op->getOperand(0);
            ValuePtr bVal = op->getOperand(1);
            ValuePtr dA = builder.createMul(dOut, bVal);
            ValuePtr dB = builder.createMul(dOut, aVal);
            accumulateAdjoint(aVal, reduceBroadcast(dA, getShape(aVal)));
            accumulateAdjoint(bVal, reduceBroadcast(dB, getShape(bVal)));
        } else if (opName == "div") {
            ValuePtr aVal = op->getOperand(0);
            ValuePtr bVal = op->getOperand(1);
            ValuePtr dA = builder.createDiv(dOut, bVal);
            ValuePtr b2 = builder.createMul(bVal, bVal);
            ValuePtr negDOutA = builder.createMul(builder.createConstant(-1.0, scalarF32), builder.createMul(dOut, aVal));
            ValuePtr dB = builder.createDiv(negDOutA, b2);
            accumulateAdjoint(aVal, reduceBroadcast(dA, getShape(aVal)));
            accumulateAdjoint(bVal, reduceBroadcast(dB, getShape(bVal)));
        } else if (opName == "relu") {
            ValuePtr inVal = op->getOperand(0);
            ValuePtr dIn = builder.createMul(dOut, inVal);
            accumulateAdjoint(inVal, dIn);
        } else if (opName == "gelu") {
            ValuePtr inVal = op->getOperand(0);
            ValuePtr dIn = builder.createMul(dOut, inVal);
            accumulateAdjoint(inVal, dIn);
        } else if (opName == "exp") {
            ValuePtr inVal = op->getOperand(0);
            ValuePtr dIn = builder.createMul(dOut, outVal); // d/dx exp(x) = exp(x)
            accumulateAdjoint(inVal, dIn);
        } else if (opName == "log") {
            ValuePtr inVal = op->getOperand(0);
            ValuePtr dIn = builder.createDiv(dOut, inVal); // d/dx log(x) = 1/x
            accumulateAdjoint(inVal, dIn);
        } else if (opName == "matmul") {
            ValuePtr xVal = op->getOperand(0);
            ValuePtr wVal = op->getOperand(1);
            Shape sX = getShape(xVal);
            Shape sW = getShape(wVal);

            if (sX.size() == 2 && sW.size() == 2) {
                ValuePtr wT = builder.createTranspose(wVal, {1, 0});
                ValuePtr xT = builder.createTranspose(xVal, {1, 0});
                ValuePtr dX = builder.createMatmul(dOut, wT);
                ValuePtr dW = builder.createMatmul(xT, dOut);
                accumulateAdjoint(xVal, dX);
                accumulateAdjoint(wVal, dW);
            } else if (sX.size() == 3 && sW.size() == 3) {
                ValuePtr wT = builder.createTranspose(wVal, {0, 2, 1});
                ValuePtr xT = builder.createTranspose(xVal, {0, 2, 1});
                ValuePtr dX = builder.createMatmul(dOut, wT);
                ValuePtr dW = builder.createMatmul(xT, dOut);
                accumulateAdjoint(xVal, dX);
                accumulateAdjoint(wVal, dW);
            }
        } else if (opName == "softmax") {
            // dX = S * (dY - sum(dY * S))
            ValuePtr inVal = op->getOperand(0);
            ValuePtr prod = builder.createMul(dOut, outVal);
            ValuePtr sumProd = builder.createSum(prod, -1, true);
            ValuePtr sub = builder.createSub(dOut, sumProd);
            ValuePtr dX = builder.createMul(outVal, sub);
            accumulateAdjoint(inVal, dX);
        } else if (opName == "log_softmax") {
            // dX = dY - softmax(X) * sum(dY)
            ValuePtr inVal = op->getOperand(0);
            ValuePtr sumDOut = builder.createSum(dOut, -1, true);
            ValuePtr softmaxVal = builder.createExp(outVal);
            ValuePtr prod = builder.createMul(softmaxVal, sumDOut);
            ValuePtr dX = builder.createSub(dOut, prod);
            accumulateAdjoint(inVal, dX);
        } else if (opName == "cross_entropy") {
            // dz = (softmax(z) - onehot(y)) / N
            ValuePtr logits = op->getOperand(0);
            ValuePtr targets = op->getOperand(1);
            Shape logitShape = getShape(logits);
            int64_t numClasses = logitShape.empty() ? 10 : logitShape.back();
            ValuePtr smax = builder.createSoftmax(logits, -1);
            ValuePtr oneHot = builder.createOneHot(targets, numClasses);
            ValuePtr diff = builder.createSub(smax, oneHot);
            int64_t N = logitShape.empty() ? 1 : logitShape[0];
            ValuePtr nConst = builder.createConstant(static_cast<double>(N), scalarF32);
            ValuePtr dz = builder.createDiv(diff, nConst);
            ValuePtr scaledDz = builder.createMul(dz, dOut);
            accumulateAdjoint(logits, scaledDz);
        } else if (opName == "sum" || opName == "mean") {
            ValuePtr inVal = op->getOperand(0);
            accumulateAdjoint(inVal, dOut);
        } else if (opName == "transpose") {
            ValuePtr inVal = op->getOperand(0);
            auto perm = std::get<std::vector<int64_t>>(op->getAttribute("perm"));
            std::vector<int64_t> invPerm(perm.size());
            for (size_t i = 0; i < perm.size(); ++i) {
                invPerm[perm[i]] = static_cast<int64_t>(i);
            }
            ValuePtr dIn = builder.createTranspose(dOut, invPerm);
            accumulateAdjoint(inVal, dIn);
        } else if (opName == "reshape") {
            ValuePtr inVal = op->getOperand(0);
            ValuePtr dIn = builder.createReshape(dOut, getShape(inVal));
            accumulateAdjoint(inVal, dIn);
        }
    }

    // Re-insert return op with backward outputs (loss and parameter gradients)
    std::vector<ValuePtr> newReturns = {lossVal};
    for (const auto& arg : func.getArguments()) {
        if (adjoints.count(arg.get())) {
            newReturns.push_back(adjoints[arg.get()]);
        }
    }
    builder.createReturn(newReturns);

    return true;
}

} // namespace dlc
