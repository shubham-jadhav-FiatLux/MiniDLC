#pragma once

#include "dlc/passes/Pass.h"
#include <unordered_map>
#include <string>
#include <vector>

namespace dlc {

struct MemoryOffset {
    int64_t offsetBytes{0};
    int64_t sizeBytes{0};
};

class MemoryPlanningPass : public Pass {
public:
    // With `annotate`, each planned value's defining op records mem.offset / mem.size attributes for IR dumps.
    explicit MemoryPlanningPass(bool annotate = true) : annotate_(annotate) {}

    std::string getName() const override { return "mem-plan"; }
    bool runOnFunction(Function& func) override;

    int64_t getTotalArenaSize() const { return totalArenaSize_; }
    int64_t getUnoptimizedSize() const { return unoptimizedSize_; }
    int64_t getSavedBytes() const { return unoptimizedSize_ - totalArenaSize_; }
    const std::unordered_map<Value*, MemoryOffset>& getOffsets() const { return valueOffsets_; }

    std::string getMemoryReport() const;

private:
    struct ReportEntry {
        std::string name;
        MemoryOffset offset;
    };

    bool annotate_{true};
    std::vector<ReportEntry> reportEntries_;
    int64_t totalArenaSize_{0};
    int64_t unoptimizedSize_{0};
    std::unordered_map<Value*, MemoryOffset> valueOffsets_;
};

} // namespace dlc

