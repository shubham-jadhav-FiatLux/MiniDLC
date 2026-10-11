#include "dlc/backends/llvm/LLVMBackend.h"

namespace dlc::backends::llvm {

runtime::ExecutionPlan LLVMBackend::compile(const std::vector<loopir::KernelNest>& kernels) {
    runtime::ExecutionPlan plan;
    for (const auto& k : kernels) {
        runtime::KernelDispatch kd;
        kd.name = k.getName();
        kd.launchFn = [](runtime::ArenaAllocator& /*arena*/) {
            // Emits LLVM IRBuilder instructions and runs via LLJIT in M2
        };
        plan.addKernel(std::move(kd));
    }
    return plan;
}

} // namespace dlc::backends::llvm
