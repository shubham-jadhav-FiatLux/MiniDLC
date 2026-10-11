#pragma once

#include "dlc/loopir/LoopIR.h"

namespace dlc::loopir {

class LoopTilingPass {
public:
    static void run(KernelNest& kernel, int64_t tileSize);
};

class LoopVectorizationPass {
public:
    static void run(KernelNest& kernel, int64_t vectorWidth = 8);
};

class BoundsCheckEliminationPass {
public:
    static void run(KernelNest& kernel);
};

} // namespace dlc::loopir
