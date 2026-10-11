#pragma once

#include "dlc/passes/Pass.h"
#include <unordered_map>

namespace dlc {

struct MemoryOffset {
    int64_t offsetBytes{0};
    int64_t sizeBytes{0};
};

class MemoryPlanningPass : public Pass {
public:
    std::string getName() const override { return "mem-plan"; }
    bool runOnFunction(Function& func) override;

    int64_t getTotalArenaSize() const { return totalArenaSize_; }
    const std::unordered_map<Value*, MemoryOffset>& getOffsets() const { return valueOffsets_; }

private:
    int64_t totalArenaSize_{0};
    std::unordered_map<Value*, MemoryOffset> valueOffsets_;
};

} // namespace dlc
