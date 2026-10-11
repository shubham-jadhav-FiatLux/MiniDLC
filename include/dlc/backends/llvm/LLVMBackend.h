#pragma once

#include "dlc/backends/Backend.h"

namespace dlc::backends::llvm {

class LLVMBackend : public Backend {
public:
    std::string getName() const override { return "llvm-cpu"; }
    runtime::ExecutionPlan compile(const std::vector<loopir::KernelNest>& kernels) override;
};

} // namespace dlc::backends::llvm
