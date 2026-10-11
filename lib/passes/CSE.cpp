#include "dlc/passes/CSE.h"
#include <unordered_map>
#include <sstream>

namespace dlc {

namespace {
std::string getOpKey(const Operation& op) {
    std::ostringstream oss;
    oss << op.getOpName() << ":";
    for (size_t i = 0; i < op.getNumOperands(); ++i) {
        auto operand = op.getOperand(i);
        oss << (operand ? std::to_string(operand->getId()) : "null") << ",";
    }
    return oss.str();
}
} // anonymous namespace

bool CSEPass::runOnFunction(Function& func) {
    bool changed = false;
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    std::unordered_map<std::string, ValuePtr> exprMap;
    std::vector<Operation*> toRemove;

    for (const auto& opPtr : entry->getOperations()) {
        Operation* op = opPtr.get();
        if (op->getOpName() == "return" || op->getNumResults() == 0) {
            continue;
        }

        std::string key = getOpKey(*op);
        auto it = exprMap.find(key);
        if (it != exprMap.end()) {
            // Found duplicate: replace results with existing value
            ValuePtr existingVal = it->second;
            ValuePtr currentVal = op->getResult(0);

            auto uses = currentVal->getUses();
            for (auto* user : uses) {
                for (size_t i = 0; i < user->getNumOperands(); ++i) {
                    if (user->getOperand(i) == currentVal) {
                        user->setOperand(i, existingVal);
                        changed = true;
                    }
                }
            }
            toRemove.push_back(op);
        } else {
            if (op->getNumResults() > 0) {
                exprMap[key] = op->getResult(0);
            }
        }
    }

    for (auto* deadOp : toRemove) {
        entry->removeOperation(deadOp);
    }

    return changed;
}

} // namespace dlc
