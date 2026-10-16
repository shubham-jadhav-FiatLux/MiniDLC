#include "dlc/passes/MemoryPlanning.h"
#include <algorithm>
#include <vector>
#include <sstream>
#include <iomanip>

namespace dlc {

struct LiveInterval {
    Value* val{nullptr};
    int64_t start{0};
    int64_t end{0};
    int64_t sizeBytes{0};
};

static bool intervalsInterfere(const LiveInterval& a, const LiveInterval& b) {
    return std::max(a.start, b.start) <= std::min(a.end, b.end);
}

bool MemoryPlanningPass::runOnFunction(Function& func) {
    valueOffsets_.clear();
    totalArenaSize_ = 0;
    unoptimizedSize_ = 0;

    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    std::unordered_map<Value*, LiveInterval> intervals;
    const auto& ops = entry->getOperations();
    int64_t totalSteps = static_cast<int64_t>(ops.size());

    // 1. Initialize function arguments (live from step 0)
    for (const auto& arg : func.getArguments()) {
        int64_t sz = 0;
        if (arg->getType()->isTensor()) {
            sz = std::static_pointer_cast<TensorType>(arg->getType())->getSizeBytes();
        }
        intervals[arg.get()] = LiveInterval{arg.get(), 0, 0, sz};
    }

    // 2. Compute live intervals across all ops
    for (int64_t step = 0; step < totalSteps; ++step) {
        Operation* op = ops[step].get();

        // Operands are read at this step, extending their live range
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            Value* operand = op->getOperand(i).get();
            if (intervals.count(operand)) {
                intervals[operand].end = std::max(intervals[operand].end, step);
            }
        }

        // Return op keeps its operands live until function completion
        if (op->getOpName() == "return") {
            for (size_t i = 0; i < op->getNumOperands(); ++i) {
                Value* operand = op->getOperand(i).get();
                if (intervals.count(operand)) {
                    intervals[operand].end = totalSteps;
                }
            }
            continue;
        }

        // Op results are born at this step
        for (size_t i = 0; i < op->getNumResults(); ++i) {
            Value* res = op->getResult(i).get();
            int64_t sz = 0;
            if (res->getType()->isTensor()) {
                sz = std::static_pointer_cast<TensorType>(res->getType())->getSizeBytes();
            }
            intervals[res] = LiveInterval{res, step, step, sz};
        }
    }

    // 3. Collect and sort intervals by size descending (greedy heuristic)
    std::vector<LiveInterval> sortedIntervals;
    for (auto& [val, interval] : intervals) {
        if (interval.sizeBytes > 0) {
            sortedIntervals.push_back(interval);
            int64_t alignedSize = (interval.sizeBytes + 63) & ~63;
            unoptimizedSize_ += alignedSize;
        }
    }
    std::sort(sortedIntervals.begin(), sortedIntervals.end(),
              [](const LiveInterval& a, const LiveInterval& b) {
                  if (a.sizeBytes != b.sizeBytes) return a.sizeBytes > b.sizeBytes;
                  return a.start < b.start;
              });

    // 4. Greedy offset assignment with interval interference checking
    struct AssignedTensor {
        LiveInterval interval;
        int64_t offset;
        int64_t alignedSize;
    };
    std::vector<AssignedTensor> assigned;

    for (const auto& item : sortedIntervals) {
        int64_t alignedSize = (item.sizeBytes + 63) & ~63;

        // Collect candidate offsets: 0 and end of every interfering assigned tensor
        std::vector<int64_t> candidates = {0};
        for (const auto& other : assigned) {
            if (intervalsInterfere(item, other.interval)) {
                candidates.push_back(other.offset + other.alignedSize);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        int64_t chosenOffset = -1;
        for (int64_t cand : candidates) {
            // Must be 64-byte aligned
            int64_t alignedCand = (cand + 63) & ~63;

            // Check conflict with all interfering assigned tensors
            bool conflict = false;
            for (const auto& other : assigned) {
                if (intervalsInterfere(item, other.interval)) {
                    int64_t aStart = alignedCand;
                    int64_t aEnd = alignedCand + alignedSize;
                    int64_t bStart = other.offset;
                    int64_t bEnd = other.offset + other.alignedSize;

                    if (std::max(aStart, bStart) < std::min(aEnd, bEnd)) {
                        conflict = true;
                        break;
                    }
                }
            }

            if (!conflict) {
                chosenOffset = alignedCand;
                break;
            }
        }

        if (chosenOffset == -1) {
            chosenOffset = totalArenaSize_;
        }

        valueOffsets_[item.val] = MemoryOffset{chosenOffset, item.sizeBytes};
        assigned.push_back(AssignedTensor{item, chosenOffset, alignedSize});
        totalArenaSize_ = std::max(totalArenaSize_, chosenOffset + alignedSize);
    }

    return true;
}

std::string MemoryPlanningPass::getMemoryReport() const {
    std::ostringstream oss;
    oss << "=== Memory Planning Report ===\n";
    oss << "Total Peak Arena Size: " << totalArenaSize_ << " bytes ("
        << (totalArenaSize_ / 1024.0 / 1024.0) << " MB)\n";
    oss << "Unoptimized Size:      " << unoptimizedSize_ << " bytes ("
        << (unoptimizedSize_ / 1024.0 / 1024.0) << " MB)\n";
    if (unoptimizedSize_ > 0) {
        double savedPct = 100.0 * (1.0 - static_cast<double>(totalArenaSize_) / unoptimizedSize_);
        oss << "Memory Saved by Reuse: " << getSavedBytes() << " bytes ("
            << std::fixed << std::setprecision(1) << savedPct << "%)\n";
    }
    oss << "Allocated Tensors:\n";
    for (const auto& [val, offset] : valueOffsets_) {
        oss << "  %" << val->getName() << " -> offset " << offset.offsetBytes
            << " bytes, size " << offset.sizeBytes << " bytes\n";
    }
    return oss.str();
}

} // namespace dlc

