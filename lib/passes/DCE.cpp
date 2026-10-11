#include "dlc/passes/DCE.h"
#include <unordered_set>
#include <vector>

namespace dlc {

bool DCEPass::runOnFunction(Function& func) {
    bool changed = false;
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    std::unordered_set<Operation*> liveOps;
    std::vector<Operation*> worklist;

    // Seeds: return operations and operations with side-effects
    for (const auto& opPtr : entry->getOperations()) {
        Operation* op = opPtr.get();
        if (op->getOpName() == "return") {
            liveOps.insert(op);
            worklist.push_back(op);
        }
    }

    // Traverse use-def backwards
    while (!worklist.empty()) {
        Operation* curr = worklist.back();
        worklist.pop_back();

        for (size_t i = 0; i < curr->getNumOperands(); ++i) {
            auto operand = curr->getOperand(i);
            if (operand && operand->getDefiningOp()) {
                Operation* defOp = operand->getDefiningOp();
                if (liveOps.insert(defOp).second) {
                    worklist.push_back(defOp);
                }
            }
        }
    }

    // Remove operations not marked live
    std::vector<Operation*> toRemove;
    for (const auto& opPtr : entry->getOperations()) {
        Operation* op = opPtr.get();
        if (liveOps.find(op) == liveOps.end()) {
            toRemove.push_back(op);
            changed = true;
        }
    }

    for (auto* deadOp : toRemove) {
        entry->removeOperation(deadOp);
    }

    return changed;
}

} // namespace dlc
