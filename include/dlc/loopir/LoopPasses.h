#pragma once

#include "dlc/loopir/LoopIR.h"

namespace dlc::loopir {

class LoopTilingPass {
public:
    static void run(KernelNest& kernel, int64_t tileM = 32, int64_t tileN = 64, int64_t tileK = 32);
};

class LoopVectorizationPass {
public:
    static void run(KernelNest& kernel, int64_t vectorWidth = 8);
};

class LoopUnrollingPass {
public:
    static void run(KernelNest& kernel, int64_t unrollFactor = 4);
};

class LoopParallelizationPass {
public:
    static void run(KernelNest& kernel);
};

class BoundsCheckEliminationPass {
public:
    static bool run(KernelNest& kernel);
};

} // namespace dlc::loopir

