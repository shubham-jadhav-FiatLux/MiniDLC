#pragma once

#include "dlc/loopir/LoopIR.h"
#include "dlc/runtime/ExecutionPlan.h"
#include <string>

namespace dlc::backends {

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string getName() const = 0;
    virtual runtime::ExecutionPlan compile(const std::vector<loopir::KernelNest>& kernels) = 0;
};

} // namespace dlc::backends
