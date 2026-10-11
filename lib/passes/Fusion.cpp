#include "dlc/passes/Fusion.h"
#include <vector>

namespace dlc {

bool FusionPass::runOnFunction(Function& func) {
    fusedGroups_.clear();
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    int64_t currentGroupId = 0;
    FusedGroup currentGroup;
    currentGroup.groupId = currentGroupId++;

    for (const auto& opPtr : entry->getOperations()) {
        Operation* op = opPtr.get();
        if (op->getOpName() == "return") continue;

        // Group elementwise ops or matmul epilogues
        if (op->getOpName() == "add" || op->getOpName() == "relu" || op->getOpName() == "mul") {
            currentGroup.operations.push_back(op);
            currentGroup.opSet.insert(op);
        } else {
            if (!currentGroup.operations.empty()) {
                fusedGroups_.push_back(std::move(currentGroup));
                currentGroup = FusedGroup();
                currentGroup.groupId = currentGroupId++;
            }
            FusedGroup standalone;
            standalone.groupId = currentGroupId++;
            standalone.operations.push_back(op);
            standalone.opSet.insert(op);
            fusedGroups_.push_back(std::move(standalone));
        }
    }

    if (!currentGroup.operations.empty()) {
        fusedGroups_.push_back(std::move(currentGroup));
    }

    return !fusedGroups_.empty();
}

} // namespace dlc
