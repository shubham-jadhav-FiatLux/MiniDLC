#pragma once

#include "dlc/loopir/LoopIR.h"
#include <string>
#include <vector>

namespace dlc::backends::llvm_emit {

// The GPU kernels compiled for NVIDIA GPUs: LLVM IR for nvptx64-nvidia-cuda, LLVM's O3 pipeline with the NVPTX
// target machine (address-space inference, among others), then the NVPTX backend's PTX assembly. The PTX is loaded
// by the CUDA driver (cuModuleLoadData), which compiles it for the GPU it runs on; nothing here needs CUDA.
// The PTX goes to the kernel cache (KernelCache.h); with `keepIR` the LLVM IR is kept too (and the cache only
// written, since the IR needs the pipeline to run).
struct PtxModule {
    std::string arch;        // e.g. "sm_75"
    std::string emittedIR;   // as emitted (keepIR)
    std::string optimizedIR; // after O3 (keepIR)
    std::string ptx;
    bool fromCache{false};
};

PtxModule compilePtx(const std::vector<loopir::Kernel>& kernels, const std::string& arch = "sm_75", bool keepIR = true);

} // namespace dlc::backends::llvm_emit
