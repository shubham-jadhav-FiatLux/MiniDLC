#pragma once

#include "dlc/loopir/LoopIR.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm::orc {
class LLJIT;
}

namespace dlc::backends::llvm_emit {

// Compiles loop kernels for the host CPU: one LLVM module, LLVM's O3 pipeline with the host's TargetMachine, then ORC
// LLJIT. Math functions (expf, tanhf, ...) resolve to the process's libm.
class CpuJit {
public:
    using KernelFn = void (*)(void** args, int64_t begin, int64_t end);

    explicit CpuJit(const std::vector<loopir::Kernel>& kernels);
    ~CpuJit();

    KernelFn function(size_t index) const { return functions_.at(index); }
    const std::string& emittedIR() const { return emitted_; }
    const std::string& optimizedIR() const { return optimized_; }
    const std::string& cpuName() const { return cpu_; }

private:
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::vector<KernelFn> functions_;
    std::string emitted_;
    std::string optimized_;
    std::string cpu_;
};

// The emitted (and, with `optimize`, O3-optimized) LLVM IR for the host CPU, without compiling it.
std::string cpuModuleText(const std::vector<loopir::Kernel>& kernels, bool optimize);

} // namespace dlc::backends::llvm_emit
