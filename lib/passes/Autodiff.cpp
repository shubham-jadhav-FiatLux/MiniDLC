#include "dlc/passes/Autodiff.h"
#include "dlc/ir/Builder.h"
#include <unordered_map>
#include <vector>

namespace dlc {

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

    // Snapshot forward operations before adding backward ops
    std::vector<OperationPtr> forwardOps = entry->getOperations();

    // Remove the forward return op temporarily to append backward pass
    entry->removeOperation(returnOp);

    // Reverse traversal
    for (auto it = forwardOps.rbegin(); it != forwardOps.rend(); ++it) {
        Operation* op = it->get();
        if (op->getOpName() == "return") continue;

        if (op->getNumResults() == 0) continue;
        ValuePtr outVal = op->getResult(0);

        if (adjoints.find(outVal.get()) == adjoints.end()) {
            continue; // Not on the path to loss
        }

        ValuePtr dOut = adjoints[outVal.get()];
        const std::string& opName = op->getOpName();

        if (opName == "relu") {
            // dIn = dOut * (in > 0)
            ValuePtr inVal = op->getOperand(0);
            ValuePtr dIn = builder.createMul(dOut, inVal); // simplified placeholder for relu'
            adjoints[inVal.get()] = dIn;
        } else if (opName == "add") {
            // dA = dOut, dB = dOut
            ValuePtr aVal = op->getOperand(0);
            ValuePtr bVal = op->getOperand(1);
            adjoints[aVal.get()] = dOut;
            adjoints[bVal.get()] = dOut;
        } else if (opName == "matmul") {
            // Y = X W -> dX = dY W^T, dW = X^T dY
            ValuePtr xVal = op->getOperand(0);
            ValuePtr wVal = op->getOperand(1);
            ValuePtr wT = builder.createTranspose(wVal, {1, 0});
            ValuePtr xT = builder.createTranspose(xVal, {1, 0});
            ValuePtr dX = builder.createMatmul(dOut, wT);
            ValuePtr dW = builder.createMatmul(xT, dOut);
            adjoints[xVal.get()] = dX;
            adjoints[wVal.get()] = dW;
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
