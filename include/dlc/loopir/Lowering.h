#pragma once

#include "dlc/ir/Function.h"
#include "dlc/loopir/LoopIR.h"
#include <vector>

namespace dlc::loopir {

class Lowering {
public:
    static std::vector<KernelNest> lowerFunction(const Function& func);
};

} // namespace dlc::loopir
