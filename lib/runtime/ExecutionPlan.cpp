#include "dlc/runtime/ExecutionPlan.h"

namespace dlc::runtime {

void ExecutionPlan::execute(ArenaAllocator& arena) {
    executionTimesMs_.clear();
    for (const auto& kernel : dispatches_) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (kernel.launchFn) {
            kernel.launchFn(arena);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        executionTimesMs_.push_back(elapsedMs);
    }
}

} // namespace dlc::runtime
