#pragma once

#include "dlc/backends/Backend.h"

namespace dlc::backends::cuda {

class CUDABackend : public Backend {
public:
    std::string getName() const override { return "cuda-gpu"; }
    runtime::ExecutionPlan compile(const std::vector<loopir::KernelNest>& kernels) override;
};

} // namespace dlc::backends::cuda
