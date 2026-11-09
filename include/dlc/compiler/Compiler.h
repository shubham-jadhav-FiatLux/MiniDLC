#pragma once

#include "dlc/loopir/Lowering.h"
#include "dlc/passes/Optimizer.h"
#include "dlc/runtime/Executable.h"
#include <memory>
#include <optional>
#include <string>

namespace dlc::compiler {

struct CompileOptions {
    // "cpu": LLVM, compiled in-process with ORC LLJIT. "interp": the loop IR interpreter (slow; for testing).
    // "gpu-sim": the GPU schedules, run by the lockstep GPU interpreter on the host (slow; for testing).
    std::string target{"cpu"};
    // With an optimizer, the function is differentiated and becomes a training step (autodiff + optimizer lowering).
    std::optional<OptimizerConfig> optimizer;
    size_t threads{0}; // 0: one per hardware thread
    // Ablations (benchmarks/ablations.py): no fusion, run-time checks on every access, untiled matmuls.
    bool fuse{true};
    loopir::LoweringOptions lowering;
    // Keep the LLVM IR (as emitted and after O3) in CompileDumps. It needs LLVM's pipeline to run, so the kernel
    // cache (KernelCache.h) is not read.
    bool llvmDumps{false};
};

// The compiler's intermediate forms, for inspection.
struct CompileDumps {
    std::string graphIR;    // after the graph pipeline, annotated with fusion and buffers
    std::string loopIR;     // every loop kernel
    std::string llvmIR;     // the emitted LLVM module (cpu, cuda; with CompileOptions::llvmDumps)
    std::string llvmIROpt;  // after LLVM's O3 pipeline (with CompileOptions::llvmDumps)
    bool fromCache{false};  // the machine code (cpu) or PTX (cuda) came from the kernel cache
    std::string costReport;
    std::string memoryReport;
};

// IR text → graph passes (shape inference, canonicalization, CSE, DCE; autodiff and optimizer lowering when
// requested) → fusion → buffer assignment → lowering with the target's schedules → code → an executable. Compiles the
// first function of the module.
std::unique_ptr<runtime::Executable> compile(const std::string& irText, const CompileOptions& options,
                                             CompileDumps* dumps = nullptr);

// The "cuda" target: the plan and the PTX of every kernel, run by a GPU runtime (python/minidlc/gpu.py drives the CUDA
// driver API). The loop kernels are kept so that a launch can also be simulated on the host (simulateLaunch), which
// tests the runtime's handling of the plan without a GPU.
struct GpuArtifact {
    runtime::Plan plan;
    std::vector<loopir::Kernel> kernels;
    std::string arch;
    std::string ptx; // empty unless requested
};

GpuArtifact compileGpu(const std::string& irText, const CompileOptions& options, CompileDumps* dumps = nullptr,
                       bool ptx = true, const std::string& arch = "sm_75");

// Runs kernel `kernel` of `artifact` (every block) with the lockstep GPU interpreter; `args` are its buffers.
void simulateLaunch(const GpuArtifact& artifact, int kernel, const std::vector<void*>& args);

// A compiled step for the host CPU that can be saved and run elsewhere without recompiling: the plan and an object
// file of its kernels (python/minidlc/cpu.py saves and loads it). It runs on any machine with the same target triple
// whose CPU has every feature the code was compiled for.
struct CpuArtifact {
    runtime::Plan plan;
    std::string object;
    std::vector<std::string> kernelNames; // in launch order
    std::string triple, cpu, features;
};

CpuArtifact compileCpu(const std::string& irText, const CompileOptions& options, CompileDumps* dumps = nullptr);

// An executable from a saved CPU artifact (IRError if this host cannot run it).
std::unique_ptr<runtime::Executable> loadCpu(const runtime::Plan& plan, const std::string& object,
                                             const std::vector<std::string>& kernelNames, const std::string& triple,
                                             const std::string& features, size_t threads = 0);

// The plan of runtime::toJson, read back (IRError if malformed).
runtime::Plan planFromJson(const std::string& json);

// Fuses, assigns buffers and lowers `func` as it is (no graph passes), for dumps: the loop kernels in launch order.
std::vector<loopir::LoweredKernel> lowerForInspection(Function& func, loopir::Target target,
                                                      const loopir::LoweringOptions& options = {});

} // namespace dlc::compiler
