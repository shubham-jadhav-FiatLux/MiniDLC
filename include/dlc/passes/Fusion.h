#pragma once

#include "dlc/passes/Pass.h"
#include <unordered_map>
#include <vector>

namespace dlc {

// What a kernel computes, which decides its lowering and schedule (P4/P5).
enum class KernelKind {
    Matmul,      // one matmul, plus an elementwise epilogue evaluated in registers
    Elementwise, // elementwise ops sharing one iteration space; any number of outputs
    Reduction,   // sum / mean / reduce_max over axes
    Row,         // softmax, log_softmax, layernorm, cross_entropy: a reduction and a pass over each row
    Gather,      // embedding, onehot
    Scatter,     // scatter_add
    Transpose,   // a permuted copy
    Fill,        // a constant that some kernel reads from memory
};

std::string kernelKindName(KernelKind kind);

struct Kernel {
    int64_t id{0};
    KernelKind kind{KernelKind::Elementwise};
    std::string name;                // unique within the function, e.g. "k3_matmul_add_relu"
    std::vector<Operation*> ops;     // members, in program order
    std::vector<Value*> inputs;      // values read from memory: arguments, reshapes, other kernels' outputs
    std::vector<Value*> outputs;     // members' results needed outside the kernel (other kernels, reshapes, results)
    Shape iterationShape;            // matmul / elementwise output shape
};

// Groups the ops of a function into kernels. Matmuls absorb an elementwise epilogue; elementwise ops with the same
// output shape merge into one kernel with any number of outputs; the other ops are kernels of their own. A merge is
// legal only if the kernel graph stays acyclic (no path from one group to the other through a third), which also
// guarantees that every input of a kernel is computed before it runs. Splat constants read by elementwise code are
// inlined; constants read from memory get a Fill kernel. Reshapes need no kernel: they alias their operand.
class FusionPass : public Pass {
public:
    // With `annotate`, each member records fusion.group / fusion.kernel attributes for IR dumps. With fuse = false,
    // every op gets its own kernel (an ablation; splat constants are still inlined).
    explicit FusionPass(bool annotate = true, bool fuse = true) : annotate_(annotate), fuse_(fuse) {}

    std::string getName() const override { return "fusion"; }
    bool runOnFunction(Function& func) override;

    // Kernels in execution order (a topological order of the kernel graph), for the last function run.
    const std::vector<Kernel>& getKernels() const { return kernels_; }
    // The kernel whose member computes `value`; nullptr for arguments, reshapes and inlined constants.
    const Kernel* kernelOf(const Value* value) const;

private:
    bool annotate_{true};
    bool fuse_{true};
    std::vector<Kernel> kernels_;
    std::unordered_map<const Value*, size_t> kernelOfValue_;
};

// Ops that fuse into elementwise kernels and matmul epilogues.
bool isElementwiseOp(const std::string& opName);

} // namespace dlc
