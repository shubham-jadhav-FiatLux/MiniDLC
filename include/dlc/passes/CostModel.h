#pragma once

#include "dlc/passes/Fusion.h"
#include <string>
#include <vector>

namespace dlc {

// Arithmetic and compulsory memory traffic: every operand is read once and every result written once. Transcendental
// functions count as one FLOP; GELU, softmax, LayerNorm and cross-entropy count the arithmetic of their formula.
struct Cost {
    double flops{0.0};
    int64_t bytesRead{0};
    int64_t bytesWritten{0};

    int64_t bytes() const { return bytesRead + bytesWritten; }
    // FLOPs per byte moved (the x axis of a roofline plot).
    double intensity() const { return bytes() == 0 ? 0.0 : flops / static_cast<double>(bytes()); }
};

Cost opCost(const Operation& op);
// A fused kernel reads its inputs and writes its outputs once; values inside it never touch memory.
Cost kernelCost(const Kernel& kernel);
int64_t tensorBytes(const Value& value);

// A table of the kernels with their FLOPs, bytes and intensity, and the totals with and without fusion.
std::string costReport(const Function& func, const std::vector<Kernel>& kernels);

} // namespace dlc
