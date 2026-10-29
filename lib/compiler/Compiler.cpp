#include "dlc/compiler/Compiler.h"
#include "dlc/backends/llvm/CpuJit.h"
#include "dlc/backends/llvm/Ptx.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/Parser.h"
#include "dlc/ir/SlotTracker.h"
#include "dlc/ir/Verifier.h"
#include "dlc/loopir/Interpreter.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/BufferAssignment.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/Canonicalize.h"
#include "dlc/passes/CostModel.h"
#include "dlc/passes/DCE.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/ShapeInference.h"

namespace dlc::compiler {

namespace {

class InterpreterRunner : public runtime::KernelRunner {
public:
    InterpreterRunner(std::vector<loopir::Kernel> kernels, std::string name) : kernels_(std::move(kernels)), name_(std::move(name)) {}

    void run(int kernel, void** args, int64_t begin, int64_t end) override {
        const loopir::Kernel& k = kernels_.at(kernel);
        std::vector<void*> buffers(args, args + k.globalBuffers().size());
        loopir::interpret(k, buffers, begin, end);
    }
    std::string name() const override { return name_; }

private:
    std::vector<loopir::Kernel> kernels_;
    std::string name_;
};

class JitRunner : public runtime::KernelRunner {
public:
    explicit JitRunner(std::unique_ptr<backends::llvm_emit::CpuJit> jit) : jit_(std::move(jit)) {}

    void run(int kernel, void** args, int64_t begin, int64_t end) override { jit_->function(kernel)(args, begin, end); }
    std::string name() const override { return "cpu"; }

private:
    std::unique_ptr<backends::llvm_emit::CpuJit> jit_;
};

runtime::Slot slotOf(const Buffer& b) { return {b.region, b.offset, b.size}; }

DataType dtypeOf(const Value& v) { return static_cast<const TensorType&>(*v.getType()).getElementType(); }
const Shape& shapeOf(const Value& v) { return static_cast<const TensorType&>(*v.getType()).getShape(); }

// The graph pipeline, fusion, buffer assignment and lowering for one target: the plan and its loop kernels.
struct Lowered {
    runtime::Plan plan;
    std::vector<loopir::Kernel> kernels;
};

Lowered lowerToPlan(const std::string& irText, const CompileOptions& options, loopir::Target target, CompileDumps* dumps) {
    ModulePtr module = IRParser::parseModule(irText);
    if (module->getFunctions().empty()) throw IRError("compile: the input has no function");
    auto input = Verifier::verify(*module);
    if (!input.success) {
        std::string message = "compile: the input failed verification:";
        for (const auto& e : input.errors) message += "\n  " + e;
        throw IRError(message);
    }
    Function& func = *module->getFunctions()[0];

    PassManager pm;
    pm.setVerifyEach(true);
    pm.addPass(std::make_unique<ShapeInferencePass>());
    pm.addPass(std::make_unique<CanonicalizePass>());
    pm.addPass(std::make_unique<CSEPass>());
    pm.addPass(std::make_unique<DCEPass>());
    if (options.optimizer) {
        pm.addPass(std::make_unique<AutodiffPass>());
        pm.addPass(std::make_unique<OptimizerPass>(*options.optimizer));
        pm.addPass(std::make_unique<CanonicalizePass>());
        pm.addPass(std::make_unique<CSEPass>());
        pm.addPass(std::make_unique<DCEPass>());
    }
    pm.run(*module);

    FusionPass fusion(/*annotate=*/dumps != nullptr, options.fuse);
    fusion.runOnFunction(func);
    const std::vector<Kernel>& kernels = fusion.getKernels();
    BufferAssignment buffers(func, kernels);
    buffers.verify();
    auto lowered = loopir::lowerFunction(func, kernels, buffers, target, options.lowering);

    // The plan: arguments and results with their slots, the launches in order, the final copies.
    runtime::Plan plan;
    SlotTracker names(func);
    plan.function = func.getName();
    plan.persistentBytes = buffers.persistentSize();
    plan.transientBytes = buffers.transientSize();
    const auto& args = func.getArguments();
    for (size_t i = 0; i < args.size(); ++i) {
        const Value& a = *args[i];
        plan.args.push_back({names.name(&a), func.getArgRole(i), dtypeOf(a), shapeOf(a), slotOf(buffers.bufferOf(&a)),
                             func.getArgInit(i)});
    }
    const Operation* ret = func.getEntryBlock()->getOperations().back().get();
    for (size_t i = 0; i < ret->getNumOperands(); ++i) {
        const Value& v = *ret->getOperand(i);
        const ResultInfo& info = func.getResultInfo()[i];
        std::string argName = info.kind == ResultKind::Value ? "" : names.name(args[info.arg].get());
        plan.results.push_back({info.kind, argName, dtypeOf(v), shapeOf(v), slotOf(buffers.bufferOf(&v))});
    }
    std::vector<loopir::Kernel> loopKernels;
    for (size_t i = 0; i < lowered.size(); ++i) {
        runtime::Launch launch;
        launch.kernel = static_cast<int>(i);
        launch.name = lowered[i].kernel.name;
        for (const Buffer* b : lowered[i].args) launch.args.push_back(slotOf(*b));
        const loopir::Kernel& lk = lowered[i].kernel;
        // A simulated GPU launch runs its blocks across the host threads.
        launch.parallelExtent = lk.gpu ? lk.grid[0] * lk.grid[1] * lk.grid[2] : lk.parallelExtent();
        launch.grid = lk.grid;
        launch.block = lk.block;
        launch.flops = lowered[i].kernel.flops;
        launch.bytes = lowered[i].kernel.bytes;
        plan.launches.push_back(std::move(launch));
        loopKernels.push_back(std::move(lowered[i].kernel));
    }
    for (const FinalCopy& c : buffers.finalCopies()) {
        plan.copies.push_back({slotOf(buffers.bufferOf(c.from)), slotOf(buffers.bufferOf(c.to))});
    }

    if (dumps) {
        dumps->graphIR = func.toString();
        dumps->costReport = costReport(func, kernels);
        dumps->memoryReport = buffers.report(func);
        for (const auto& k : loopKernels) dumps->loopIR += loopir::toString(k);
    }
    return {std::move(plan), std::move(loopKernels)};
}

} // namespace

std::unique_ptr<runtime::Executable> compile(const std::string& irText, const CompileOptions& options,
                                             CompileDumps* dumps) {
    const bool gpu = options.target == "gpu-sim";
    if (!gpu && options.target != "cpu" && options.target != "interp") {
        throw IRError("compile: unknown target '" + options.target + "' (cpu, interp or gpu-sim; 'cuda' builds a GpuArtifact)");
    }
    auto [plan, loopKernels] = lowerToPlan(irText, options, gpu ? loopir::Target::GPU : loopir::Target::CPU, dumps);

    std::unique_ptr<runtime::KernelRunner> runner;
    if (options.target == "interp" || gpu) {
        runner = std::make_unique<InterpreterRunner>(loopKernels, options.target);
    } else {
        auto jit = std::make_unique<backends::llvm_emit::CpuJit>(loopKernels);
        if (dumps) {
            dumps->llvmIR = jit->emittedIR();
            dumps->llvmIROpt = jit->optimizedIR();
        }
        runner = std::make_unique<JitRunner>(std::move(jit));
    }
    return std::make_unique<runtime::Executable>(std::move(plan), std::move(runner), options.threads);
}

GpuArtifact compileGpu(const std::string& irText, const CompileOptions& options, CompileDumps* dumps, bool ptx,
                       const std::string& arch) {
    auto [plan, kernels] = lowerToPlan(irText, options, loopir::Target::GPU, dumps);
    GpuArtifact out;
    out.plan = std::move(plan);
    out.kernels = std::move(kernels);
    out.arch = arch;
    if (ptx) {
        auto module = backends::llvm_emit::compilePtx(out.kernels, arch);
        out.ptx = std::move(module.ptx);
        if (dumps) {
            dumps->llvmIR = std::move(module.emittedIR);
            dumps->llvmIROpt = std::move(module.optimizedIR);
        }
    }
    return out;
}

void simulateLaunch(const GpuArtifact& artifact, int kernel, const std::vector<void*>& args) {
    const loopir::Kernel& k = artifact.kernels.at(kernel);
    const auto& g = k.grid;
    loopir::interpret(k, args, 0, g[0] * g[1] * g[2]);
}

std::vector<loopir::LoweredKernel> lowerForInspection(Function& func, loopir::Target target) {
    FusionPass fusion(/*annotate=*/false);
    fusion.runOnFunction(func);
    BufferAssignment buffers(func, fusion.getKernels());
    buffers.verify();
    auto lowered = loopir::lowerFunction(func, fusion.getKernels(), buffers, target);
    for (auto& l : lowered) l.args.clear(); // the buffers belong to this call's assignment
    return lowered;
}

} // namespace dlc::compiler
