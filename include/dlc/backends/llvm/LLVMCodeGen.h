#pragma once

#include "dlc/loopir/LoopIR.h"
#include <string>
#include <vector>

namespace dlc::backends::llvm {

class LLVMCodeGen {
public:
    // Emits standard textual LLVM IR for a list of Loop IR kernel nests
    static std::string generateModuleIR(const std::vector<loopir::KernelNest>& kernels);

    // Emits LLVM IR for an individual kernel
    static std::string generateKernelIR(const loopir::KernelNest& kernel);
};

} // namespace dlc::backends::llvm
