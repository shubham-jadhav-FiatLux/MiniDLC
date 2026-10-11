#include "dlc/passes/MemoryPlanning.h"
#include <algorithm>
#include <vector>

namespace dlc {

struct LiveInterval {
    Value* val;
    int64_t start;
    int64_t end;
    int64_t sizeBytes;
};

bool MemoryPlanningPass::runOnFunction(Function& func) {
    valueOffsets_.clear();
    totalArenaSize_ = 0;

    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    std::unordered_map<Value*, LiveInterval> intervals;
    const auto& ops = entry->getOperations();

    // Compute intervals
    for (int64_t step = 0; step < static_cast<int64_t>(ops.size()); ++step) {
        Operation* op = ops[step].get();
        for (size_t i = 0; i < op->getNumResults(); ++i) {
            Value* res = op->getResult(i).get();
            int64_t sz = 0;
            if (res->getType()->isTensor()) {
                sz = std::static_pointer_cast<TensorType>(res->getType())->getSizeBytes();
            }
            intervals[res] = LiveInterval{res, step, step, sz};
        }

        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            Value* operand = op->getOperand(i).get();
            if (intervals.count(operand)) {
                intervals[operand].end = step;
            }
        }
    }

    // Greedy offset assignment
    std::vector<LiveInterval> sortedIntervals;
    for (auto& [val, interval] : intervals) {
        sortedIntervals.push_back(interval);
    }
    // Sort by largest tensor size first
    std::sort(sortedIntervals.begin(), sortedIntervals.end(),
              [](const LiveInterval& a, const LiveInterval& b) {
                  return a.sizeBytes > b.sizeBytes;
              });

    int64_t currentOffset = 0;
    for (const auto& item : sortedIntervals) {
        valueOffsets_[item.val] = MemoryOffset{currentOffset, item.sizeBytes};
        // 64-byte alignment
        int64_t alignedSize = (item.sizeBytes + 63) & ~63;
        currentOffset += alignedSize;
    }

    totalArenaSize_ = currentOffset;
    return true;
}

} // namespace dlc
