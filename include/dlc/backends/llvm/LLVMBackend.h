#pragma once

#include "dlc/backends/Backend.h"
#include "dlc/runtime/ThreadPool.h"
#include <string>
#include <memory>

namespace dlc::backends::llvm {

class LLVMBackend : public Backend {
public:
    explicit LLVMBackend(size_t threads = 4);
    ~LLVMBackend() override;

    std::string getName() const override { return "llvm-cpu"; }
    runtime::ExecutionPlan compile(const std::vector<loopir::KernelNest>& kernels) override;

    const std::string& getLastGeneratedIR() const { return lastGeneratedIR_; }
    runtime::ThreadPool& getThreadPool() { return threadPool_; }

private:
    std::string lastGeneratedIR_;
    runtime::ThreadPool threadPool_;
    std::vector<void*> dlHandles_;
    std::vector<std::string> tempSoFiles_;
};

} // namespace dlc::backends::llvm

