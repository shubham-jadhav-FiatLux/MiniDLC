#pragma once

#include "dlc/ir/Function.h"
#include "dlc/passes/BufferAssignment.h"
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace dlc::runtime {

// Where a buffer lives: an offset into the persistent region or the transient arena.
struct Slot {
    Region region{Region::Transient};
    int64_t offset{0};
    int64_t bytes{0};
};

struct PlanArg {
    std::string name;
    ArgRole role{ArgRole::Input};
    DataType dtype{DataType::Float32};
    Shape shape;
    Slot slot;
    std::optional<double> init; // state arguments start from it
};

struct PlanResult {
    ResultKind kind{ResultKind::Value};
    std::string arg; // the donated argument
    DataType dtype{DataType::Float32};
    Shape shape;
    Slot slot;
};

// One kernel launch: the kernel, the slot of each of its global buffers, and how its outermost parallel loop splits
// (CPU) or its launch configuration (GPU, in the kernel).
struct Launch {
    int kernel{0};
    std::string name;
    std::vector<Slot> args;
    int64_t parallelExtent{1};
    std::array<int64_t, 3> grid{1, 1, 1}; // GPU launch configuration
    std::array<int64_t, 3> block{1, 1, 1};
    double flops{0.0};
    int64_t bytes{0};
};

struct Copy {
    Slot from;
    Slot to;
};

// Everything needed to run a compiled step except the kernels' code: memory sizes, the arguments and results with
// their slots, the launches in order and the copies after them.
struct Plan {
    std::string function;
    int64_t persistentBytes{0};
    int64_t transientBytes{0};
    std::vector<PlanArg> args;
    std::vector<PlanResult> results;
    std::vector<Launch> launches;
    std::vector<Copy> copies;
};

// The plan as JSON, for runtimes outside this process (the CUDA runtime in python/minidlc/gpu.py): every field above;
// slots as {"region": "persistent"|"transient", "offset", "bytes"}.
std::string toJson(const Plan& plan);

} // namespace dlc::runtime
