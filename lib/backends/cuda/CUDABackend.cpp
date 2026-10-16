#include "dlc/backends/cuda/CUDABackend.h"
#include "dlc/backends/cuda/CUDACodeGen.h"

namespace dlc::backends::cuda {

runtime::ExecutionPlan CUDABackend::compile(const std::vector<loopir::KernelNest>& kernels) {
    runtime::ExecutionPlan plan;

    // Generate complete CUDA module
    std::string cudaModule = CUDACodeGen::generateModuleCode(kernels);

    for (const auto& k : kernels) {
        runtime::KernelDispatch kd;
        kd.name = k.getName();
        CUDALaunchConfig cfg = CUDACodeGen::calculateLaunchConfig(k);

        kd.launchFn = [cfg](runtime::ArenaAllocator& /*arena*/) {
            // Kernel launch dispatched via CUDA Driver API or NVRTC
        };
        plan.addKernel(std::move(kd));
    }
    return plan;
}

} // namespace dlc::backends::cuda
