#include "dlc/loopir/Lowering.h"
#include "dlc/ir/Ops.h"

namespace dlc::loopir {

std::vector<KernelNest> Lowering::lowerFunction(const Function& func) {
    std::vector<KernelNest> kernels;
    const Block* entry = func.getEntryBlock();
    if (!entry) return kernels;

    for (const auto& opPtr : entry->getOperations()) {
        Operation* op = opPtr.get();
        if (op->getOpName() == "return") continue;

        KernelNest kernel(op->getOpName() + "_kernel");

        if (op->getOpName() == "matmul" && op->getNumOperands() >= 2) {
            auto lhsType = std::static_pointer_cast<TensorType>(op->getOperand(0)->getType());
            auto rhsType = std::static_pointer_cast<TensorType>(op->getOperand(1)->getType());
            int64_t M = lhsType->getShape()[0];
            int64_t K = lhsType->getShape()[1];
            int64_t N = rhsType->getShape()[1];

            auto loopI = std::make_shared<ForLoop>("i", 0, M);
            auto loopJ = std::make_shared<ForLoop>("j", 0, N);
            auto loopK = std::make_shared<ForLoop>("k", 0, K);

            loopJ->addBodyStmt(loopK);
            loopI->addBodyStmt(loopJ);
            kernel.addRootStmt(loopI);
        } else if (op->getNumResults() > 0 && op->getResult(0)->getType()->isTensor()) {
            auto resType = std::static_pointer_cast<TensorType>(op->getResult(0)->getType());
            const auto& shape = resType->getShape();
            if (!shape.empty()) {
                auto outerLoop = std::make_shared<ForLoop>("d0", 0, shape[0]);
                auto current = outerLoop;
                for (size_t d = 1; d < shape.size(); ++d) {
                    auto inner = std::make_shared<ForLoop>("d" + std::to_string(d), 0, shape[d]);
                    current->addBodyStmt(inner);
                    current = inner;
                }
                kernel.addRootStmt(outerLoop);
            }
        }

        kernels.push_back(std::move(kernel));
    }

    return kernels;
}

} // namespace dlc::loopir
