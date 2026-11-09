#pragma once

#include "dlc/loopir/LoopIR.h"
#include <array>
#include <string>
#include <vector>

// Loop passes for GPU kernels: the resource check, bank-conflict and coalescing analysis, and padding of shared
// tiles against bank conflicts.
namespace dlc::loopir {

// What one block may use on the target. The defaults are sm_75 (T4) — and every architecture since sm_30 for these
// limits — with static shared memory (48 KiB; more needs dynamic shared memory and an opt-in).
struct GpuLimits {
    int64_t maxThreadsPerBlock{1024};
    std::array<int64_t, 3> maxBlock{1024, 1024, 64};
    std::array<int64_t, 3> maxGrid{2147483647, 65535, 65535};
    int64_t maxSharedBytes{48 * 1024};
    int64_t maxRegistersPerThread{255};
    int64_t registersPerBlock{64 * 1024};
};

struct GpuResources {
    int64_t threads{0};
    int64_t sharedBytes{0};
    // 32-bit words of register buffers per thread: a lower bound on the registers a thread needs (ptxas reports the
    // real count; tests/python/test_gpu_backend.py checks it with ptxas when available).
    int64_t registerWords{0};
};

// Checks the launch configuration and the block's resources against `limits` (IRError if one is exceeded).
GpuResources checkResources(const Kernel& kernel, const GpuLimits& limits = {});

// Per access site of a GPU kernel (a load or a store, in shared or global memory), what its warp requests look like,
// from the warp probe (probeWarpAccesses): every warp of block 0, the first iterations of every loop.
struct AccessStats {
    int buffer{-1};
    bool store{false};
    std::string index;   // the index expression
    int64_t requests{0}; // warp requests observed
    // Shared memory: n-way bank conflicts (32 banks of 4-byte words; lanes reading one word share it): the worst and
    // the mean over requests. 1 = conflict-free.
    int maxWays{1};
    double meanWays{1.0};
    // Global memory: 32-byte sectors per request, and the fraction of the fetched bytes that the lanes asked for
    // (1 = fully coalesced).
    double sectors{0.0};
    double efficiency{1.0};
};

std::vector<AccessStats> analyzeAccesses(const Kernel& kernel);

// Bank-conflict padding. A two-dimensional shared buffer `X` whose accesses use the row pitch through a variable
// `X_pitch` (bound by a Let) gets the pitch, from its row length up, with the fewest bank conflicts over all its
// accesses (the smallest such pitch); the pitch is substituted into the indices and the buffer grows to match.
struct PaddingDecision {
    std::string buffer;
    int64_t from{0}; // the row length
    int64_t to{0};   // the chosen pitch
    int waysBefore{1};
    int waysAfter{1};
};

std::vector<PaddingDecision> padSharedBuffers(Kernel& kernel, const GpuLimits& limits = {});

// The GPU section of the loop report: resources, padding decisions and (when analyzed) access statistics.
struct GpuReport {
    GpuResources resources;
    std::vector<PaddingDecision> padding;
    std::vector<AccessStats> accesses; // empty unless analyzed
};

std::string toString(const GpuReport& report, const Kernel& kernel);

} // namespace dlc::loopir
