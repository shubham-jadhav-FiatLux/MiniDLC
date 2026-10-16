#pragma once

#include "dlc/loopir/LoopIR.h"
#include <string>
#include <vector>

namespace dlc::backends::cuda {

struct CUDALaunchConfig {
    int gridX{1};
    int gridY{1};
    int gridZ{1};
    int blockX{1};
    int blockY{1};
    int blockZ{1};
    int sharedMemBytes{0};
};

class CUDACodeGen {
public:
    // Emits complete, compilable CUDA C++ source code for a list of loop IR kernels
    static std::string generateModuleCode(const std::vector<loopir::KernelNest>& kernels);

    // Emits CUDA C++ source code for an individual kernel
    static std::string generateKernelCode(const loopir::KernelNest& kernel);

    // Determines optimal CUDA grid and block launch dimensions for a kernel
    static CUDALaunchConfig calculateLaunchConfig(const loopir::KernelNest& kernel);
};

} // namespace dlc::backends::cuda
