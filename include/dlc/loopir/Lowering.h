#pragma once

#include "dlc/loopir/Analysis.h"
#include "dlc/loopir/LoopIR.h"
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
    std::string graphKernel; // the fused kernel it comes from
    BoundsReport bounds;     // what the loop passes proved, guarded and eliminated
};

// Lowers the fused kernels of `func` (in schedule order) into loop kernels with the target's schedules. A fused kernel
// may become several loop kernels (e.g. a reduction over leading axes: an initialization, then an accumulation).
std::vector<LoweredKernel> lowerFunction(const Function& func, const std::vector<dlc::Kernel>& kernels,
                                         const BufferAssignment& buffers, Target target);

} // namespace dlc::loopir
