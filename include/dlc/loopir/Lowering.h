#pragma once

#include "dlc/ir/Function.h"
#include "dlc/loopir/LoopIR.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/MemoryPlanning.h"
#include <vector>

namespace dlc::loopir {

class Lowering {
public:
    static std::vector<KernelNest> lowerFunction(
        const Function& func,
        const FusionPass* fusion = nullptr,
        const MemoryPlanningPass* memPlan = nullptr
    );
};

} // namespace dlc::loopir

