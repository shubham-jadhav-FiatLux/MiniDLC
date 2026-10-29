#pragma once

#include "dlc/loopir/LoopIR.h"
#include <vector>

namespace dlc::loopir {

// Executes a kernel one iteration at a time, whatever the loop kinds; `args[i]` is the memory of the i-th global
// buffer. Every load and store is bounds-checked (IRError on a violation), so it doubles as a checker for lowering
// and schedules. With the CPU calling convention, only iterations [begin, end) of an outermost Parallel loop run.
// A GPU kernel (`kernel.gpu`) runs blocks [begin, end) of its grid (linear index, x fastest), the threads of each block
// in lockstep: barriers must not diverge, shuffles must read active lanes, and shared memory starts as NaN.
void interpret(const Kernel& kernel, const std::vector<void*>& args);
void interpret(const Kernel& kernel, const std::vector<void*>& args, int64_t begin, int64_t end);

} // namespace dlc::loopir
