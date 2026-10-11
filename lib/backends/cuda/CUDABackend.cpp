#include "dlc/backends/cuda/CUDABackend.h"

namespace dlc::backends::cuda {

runtime::ExecutionPlan CUDABackend::compile(const std::vector<loopir::KernelNest>& kernels) {
    runtime::ExecutionPlan plan;
    for (const auto& k : kernels) {
        runtime::KernelDispatch kd;
        kd.name = k.getName();
        kd.launchFn = [](runtime::ArenaAllocator& /*arena*/) {
            // Emits CUDA C++ and compiles via NVRTC in M3
        };
        plan.addKernel(std::move(kd));
    }
    return plan;
}

} // namespace dlc::backends::cuda
