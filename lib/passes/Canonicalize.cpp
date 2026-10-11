#include "dlc/passes/Canonicalize.h"
#include "dlc/ir/Ops.h"

namespace dlc {

bool CanonicalizePass::runOnFunction(Function& func) {
    bool changed = false;
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    // Pattern simplifications: x + 0 -> x, x * 1 -> x
    for (auto& op : entry->getOperations()) {
        if (op->getOpName() == "add" && op->getNumOperands() == 2) {
            auto rhs = op->getOperand(1);
            if (rhs && rhs->getDefiningOp() && rhs->getDefiningOp()->getOpName() == "constant") {
                auto constOp = static_cast<ConstantOp*>(rhs->getDefiningOp());
                if (constOp->getValue() == 0.0) {
                    // Replace uses of op->getResult(0) with lhs
                    auto lhs = op->getOperand(0);
                    auto res = op->getResult(0);
                    auto uses = res->getUses();
                    for (auto* user : uses) {
                        for (size_t i = 0; i < user->getNumOperands(); ++i) {
                            if (user->getOperand(i) == res) {
                                user->setOperand(i, lhs);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    return changed;
}

} // namespace dlc
