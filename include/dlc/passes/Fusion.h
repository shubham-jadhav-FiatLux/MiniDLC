#pragma once

#include "dlc/passes/Pass.h"
#include <unordered_set>

namespace dlc {

struct FusedGroup {
    int64_t groupId;
    std::vector<Operation*> operations;
    std::unordered_set<Operation*> opSet;
};

class FusionPass : public Pass {
public:
    std::string getName() const override { return "fusion"; }
    bool runOnFunction(Function& func) override;

    const std::vector<FusedGroup>& getFusedGroups() const { return fusedGroups_; }

private:
    std::vector<FusedGroup> fusedGroups_;
};

} // namespace dlc
