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

// Compiles loop kernels for the host CPU: one LLVM module, LLVM's O3 pipeline with the host's TargetMachine, code
// generation to an object file, then ORC LLJIT links it into the process. Math functions (expf, tanhf, ...) resolve to
// the process's libm. The object file goes to the kernel cache (KernelCache.h); compiling the same kernels for the
// same CPU again loads it from there and skips LLVM. With `keepIR`, the emitted and optimized LLVM IR are kept for
// inspection (which needs the pipeline to run, so the cache is only written, not read).
// Machine code for the host CPU: an object file of the kernels, and what it was compiled for.
struct CpuObject {
    std::string object;   // ELF (or the host's format) object file bytes
    std::string triple;   // target triple
    std::string cpu;      // CPU name, e.g. "skylake"
    std::string features; // the target's feature string, e.g. "+avx2,+fma,..."
    bool fromCache{false};
    std::string emittedIR;   // with keepIR
    std::string optimizedIR; // with keepIR
};

// Emits, optimizes (O3) and compiles `kernels` for the host CPU, or takes the object from the kernel cache.
CpuObject compileCpuObject(const std::vector<loopir::Kernel>& kernels, bool keepIR = false);

// Why the host cannot run code compiled for `triple` with `features` (a different triple, or a feature the host CPU
// lacks), or "" if it can.
std::string hostIncompatibility(const std::string& triple, const std::string& features);

class CpuJit {
public:
    using KernelFn = void (*)(void** args, int64_t begin, int64_t end);

    explicit CpuJit(const std::vector<loopir::Kernel>& kernels, bool keepIR = false);
    // Links an object file compiled before (a CPU artifact); `names` are its kernels, in launch order.
    CpuJit(const std::string& object, const std::vector<std::string>& names);
    ~CpuJit();

    KernelFn function(size_t index) const { return functions_.at(index); }
    const std::string& emittedIR() const { return emitted_; }
    const std::string& optimizedIR() const { return optimized_; }
    const std::string& cpuName() const { return cpu_; }
    bool fromCache() const { return fromCache_; }

private:
    void link(const std::string& object, const std::vector<std::string>& names);

    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::vector<KernelFn> functions_;
    std::string emitted_;
    std::string optimized_;
    std::string cpu_;
    bool fromCache_{false};
};

// The emitted (and, with `optimize`, O3-optimized) LLVM IR for the host CPU, without compiling it.
std::string cpuModuleText(const std::vector<loopir::Kernel>& kernels, bool optimize);

} // namespace dlc::backends::llvm_emit
