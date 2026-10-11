#pragma once

#include "dlc/runtime/Arena.h"
#include <string>
#include <vector>
#include <functional>
#include <chrono>

namespace dlc::runtime {

struct KernelDispatch {
    std::string name;
    std::function<void(ArenaAllocator&)> launchFn;
    int64_t bytesMoved{0};
    int64_t totalFlops{0};
};

class ExecutionPlan {
public:
    ExecutionPlan() = default;

    void addKernel(KernelDispatch dispatch) {
        dispatches_.push_back(std::move(dispatch));
    }

    void execute(ArenaAllocator& arena);

    const std::vector<double>& getExecutionTimesMs() const { return executionTimesMs_; }

private:
    std::vector<KernelDispatch> dispatches_;
    std::vector<double> executionTimesMs_;
};

} // namespace dlc::runtime
