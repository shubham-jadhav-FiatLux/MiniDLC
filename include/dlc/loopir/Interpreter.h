#pragma once

#include "dlc/loopir/LoopIR.h"
#include <atomic>
#include <memory>
#include <vector>

namespace dlc::loopir {

// Race detection across the blocks of one simulated GPU launch. Blocks may run in chunks on several host threads;
// each block, when it finishes, records for every global element it touched whether it read, wrote or atomically
// added to it, and an element that one block writes (or atomically updates) and another block touches in a
// conflicting way is a race: blocks of one launch are unordered.
class GpuLaunchCheck {
public:
    explicit GpuLaunchCheck(const Kernel& kernel);
    // `flags`: kRead | kWrite | kAtomic. Throws IRError if another block's accesses to the element conflict.
    void merge(int buffer, int64_t index, uint8_t flags, int64_t block);

    static constexpr uint8_t kRead = 1, kWrite = 2, kAtomic = 4;

private:
    const Kernel& kernel_;
    std::vector<std::vector<std::atomic<uint8_t>>> flags_; // per buffer (empty for shared and register buffers)
};

// Executes a kernel one iteration at a time, whatever the loop kinds; `args[i]` is the memory of the i-th global
// buffer. Every load and store is bounds-checked (IRError on a violation), so it doubles as a checker for lowering
// and schedules. With the CPU calling convention, only iterations [begin, end) of an outermost Parallel loop run.
//
// A GPU kernel (`kernel.gpu`) runs blocks [begin, end) of its grid (linear index, x fastest), the threads of each block
// in lockstep: barriers must not diverge, shuffles must read active lanes, and shared memory starts as NaN. Because
// lockstep execution hides missing synchronization (every statement finishes for all threads before the next), the
// interpreter also checks every access for races, as a GPU would be entitled to reorder them:
//   - shared memory: two threads touching an element between the same two barriers, at least one of them writing;
//   - global memory: two threads of the launch touching an element, at least one of them writing, unless both add to
//     it atomically. (Stricter than CUDA within a block, where a barrier also orders global memory: no schedule
//     communicates through global memory inside a launch.) Conflicts between blocks go through `launch`, which must
//     be shared by every call that runs a block of the same launch; without one, the call checks its own blocks.
void interpret(const Kernel& kernel, const std::vector<void*>& args);
void interpret(const Kernel& kernel, const std::vector<void*>& args, int64_t begin, int64_t end,
               GpuLaunchCheck* launch = nullptr);

// One memory request of a warp, as the warp probe observes it: the access site (a Load expression or a Store or
// AtomicAdd statement), its buffer, and the element index of each active lane.
struct WarpRequest {
    const void* site{nullptr};
    int buffer{-1};
    bool store{false};
    int warp{0};
    std::vector<int> lanes;
    std::vector<int64_t> indices; // one per entry of `lanes`
};

// Runs every warp of block 0 of a GPU kernel on its own with symbolic memory (loads read 0, stores are dropped) and at
// most `iterations` iterations of each loop, and returns the requests the warps make to shared and global memory: the
// input of bank-conflict and coalescing analysis (GpuPasses.h). Indices that depend on loaded data see 0.
std::vector<WarpRequest> probeWarpAccesses(const Kernel& kernel, int64_t iterations = 2);

} // namespace dlc::loopir
