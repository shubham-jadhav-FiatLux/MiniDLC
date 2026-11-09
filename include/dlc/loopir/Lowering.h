#pragma once

#include "dlc/loopir/Analysis.h"
#include "dlc/loopir/GpuPasses.h"
#include "dlc/loopir/LoopIR.h"
#include "dlc/loopir/Transforms.h"
#include "dlc/passes/BufferAssignment.h"
#include "dlc/passes/Fusion.h"
#include <vector>

namespace dlc::loopir {

enum class Target { CPU, GPU };

std::string targetName(Target target);

// A loop kernel and the planned buffer behind each of its global buffers (in order). Values that share storage — a
// reshape and its operand, an in-place update and its parameter — share one global buffer.
struct LoweredKernel {
    Kernel kernel;
    std::vector<const Buffer*> args;
    std::string graphKernel;   // the fused kernel it comes from
    BoundsReport bounds;       // what the loop passes proved, guarded and eliminated
    SimplifyReport simplified; // what the simplification pass changed
    GpuReport gpu;             // GPU kernels: resources, padding, and (if analyzed) memory access statistics
};

// Lowers the fused kernels of `func` (in schedule order) into loop kernels with the target's schedules. A fused kernel
// may become several loop kernels (e.g. a reduction over leading axes: an initialization, then an accumulation).
// Then the loop passes, per kernel: (GPU) bank-conflict padding of shared tiles; guard insertion and bounds-check
// elimination; the dependence check; simplification; (GPU) the resource check.
// Switches for ablations and reports; the defaults are the compiler's normal behaviour.
struct LoweringOptions {
    bool checkEveryAccess{false}; // run-time checks on every access instead of proofs (checkEveryAccess)
    bool naiveMatmul{false};      // CPU: one scalar dot product per output instead of register tiles
    bool simplify{true};          // the loop simplification pass (ablation: off)
    bool analyzeAccesses{false};  // GPU: bank-conflict and coalescing statistics in LoweredKernel::gpu (for reports)
    GpuLimits limits;             // GPU: the resource check's limits
};

std::vector<LoweredKernel> lowerFunction(const Function& func, const std::vector<dlc::Kernel>& kernels,
                                         const BufferAssignment& buffers, Target target,
                                         const LoweringOptions& options = {});

} // namespace dlc::loopir
