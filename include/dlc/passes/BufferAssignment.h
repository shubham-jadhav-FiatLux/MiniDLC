#pragma once

#include "dlc/passes/Fusion.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace dlc {

// Persistent buffers (params, optimizer state, consts) are allocated once and live across steps; transient buffers
// (inputs, kernel outputs, results) share one arena, reused according to liveness over the kernel schedule.
enum class Region { Persistent, Transient };

std::string regionName(Region region);

struct Buffer {
    const Value* value{nullptr}; // the value the buffer is allocated for
    Region region{Region::Transient};
    int64_t offset{0};           // bytes from the start of the region
    int64_t size{0};             // bytes
    int64_t begin{0};            // live from kernel `begin` through kernel `end`; -1 is before the first kernel and
    int64_t end{0};              // K (the kernel count) after the last one
};

// After the last kernel, copy `from` into the buffer of argument `to`: a donated update that could not be done in
// place.
struct FinalCopy {
    const Value* from{nullptr};
    const Value* to{nullptr};
};

class BufferAssignment {
public:
    static constexpr int64_t kAlignment = 256;

    BufferAssignment(const Function& func, const std::vector<Kernel>& kernels);

    // The buffer that holds `value`: its own, or the one it aliases (a reshape shares its operand's buffer; an
    // in-place update shares its argument's).
    const Buffer& bufferOf(const Value* value) const;
    bool hasBuffer(const Value* value) const;
    // Whether `value` writes into the buffer of the argument it donates to.
    bool updatesInPlace(const Value* value) const;

    const std::vector<Buffer>& buffers() const { return buffers_; }
    const std::vector<FinalCopy>& finalCopies() const { return finalCopies_; }
    int64_t persistentSize() const { return persistentSize_; }
    int64_t transientSize() const { return transientSize_; }
    // The transient arena without any reuse: every transient buffer at its own offset.
    int64_t transientUnshared() const { return transientUnshared_; }

    // Throws IRError unless every two buffers that are live at the same time are disjoint, and every in-place update
    // is safe: every other reader of the argument runs before the kernel that overwrites it, which reads it only at
    // the element it writes (an elementwise kernel, or a matmul epilogue).
    void verify() const;

    std::string report(const Function& func) const;

private:
    const Value* root(const Value* value) const;

    const Function& func_;
    const std::vector<Kernel>& kernels_;
    std::vector<Buffer> buffers_;
    std::unordered_map<const Value*, size_t> bufferIndex_;
    std::unordered_map<const Value*, const Value*> aliasOf_; // value → the value whose buffer it uses
    std::unordered_map<const Value*, const Value*> inPlace_; // donated value → argument it overwrites
    std::unordered_map<const Value*, int64_t> producer_;     // value → kernel index
    std::vector<FinalCopy> finalCopies_;
    int64_t persistentSize_{0};
    int64_t transientSize_{0};
    int64_t transientUnshared_{0};
};

// Runs fusion (without annotations) for the kernel schedule, assigns and verifies the buffers, and annotates each
// kernel output's op with mem.region / mem.offset / mem.size.
class BufferAssignmentPass : public Pass {
public:
    explicit BufferAssignmentPass(bool annotate = true) : annotate_(annotate) {}

    std::string getName() const override { return "buffer-assign"; }
    bool runOnFunction(Function& func) override;

    const std::string& getReport() const { return report_; }

private:
    bool annotate_{true};
    std::string report_;
};

} // namespace dlc
