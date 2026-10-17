#pragma once

#include "dlc/passes/Pass.h"
#include <unordered_set>
#include <vector>
#include <string>

namespace dlc {

enum class FusedGroupKind {
    MatmulEpilogue,
    Elementwise,
    Reduction,
    Generic
};

struct FusedGroup {
    int64_t groupId{0};
    FusedGroupKind kind{FusedGroupKind::Generic};
    std::string name;
    std::vector<Operation*> operations;
    std::unordered_set<Operation*> opSet;

    // Matmul epilogue ops
    Operation* matmulOp{nullptr};
    Operation* biasAddOp{nullptr};
    Operation* actOp{nullptr};
};

class FusionPass : public Pass {
public:
    // With `annotate`, each grouped op records fusion.group / fusion.kernel attributes for IR dumps.
    explicit FusionPass(bool annotate = true) : annotate_(annotate) {}

    std::string getName() const override { return "fusion"; }
    bool runOnFunction(Function& func) override;

    const std::vector<FusedGroup>& getFusedGroups() const { return fusedGroups_; }

private:
    bool annotate_{true};
    std::vector<FusedGroup> fusedGroups_;
};

} // namespace dlc

