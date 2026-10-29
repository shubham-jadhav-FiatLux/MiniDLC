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
    std::string target{"cpu"};
    // With an optimizer, the function is differentiated and becomes a training step (autodiff + optimizer lowering).
    std::optional<OptimizerConfig> optimizer;
    size_t threads{0}; // 0: one per hardware thread
};

// The compiler's intermediate forms, for inspection.
struct CompileDumps {
    std::string graphIR;    // after the graph pipeline, annotated with fusion and buffers
    std::string loopIR;     // every loop kernel
    std::string llvmIR;     // the emitted LLVM module (cpu)
    std::string llvmIROpt;  // after LLVM's O3 pipeline (cpu)
    std::string costReport;
    std::string memoryReport;
};

// IR text → graph passes (shape inference, canonicalization, CSE, DCE; autodiff and optimizer lowering when
// requested) → fusion → buffer assignment → lowering with the target's schedules → code → an executable. Compiles the
// first function of the module.
std::unique_ptr<runtime::Executable> compile(const std::string& irText, const CompileOptions& options,
                                             CompileDumps* dumps = nullptr);

// Fuses, assigns buffers and lowers `func` as it is (no graph passes), for dumps: the loop kernels in launch order.
std::vector<loopir::LoweredKernel> lowerForInspection(Function& func, loopir::Target target);

} // namespace dlc::compiler
