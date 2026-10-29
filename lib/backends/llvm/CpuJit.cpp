#include "dlc/backends/llvm/CpuJit.h"
#include "dlc/backends/llvm/Emitter.h"
#include "dlc/ir/Error.h"
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mutex>

namespace dlc::backends::llvm_emit {

namespace {

void initializeLLVM() {
    static std::once_flag once;
    std::call_once(once, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    });
}

template <typename T>
T check(llvm::Expected<T> value, const char* what) {
    if (!value) throw IRError(std::string(what) + ": " + llvm::toString(value.takeError()));
    return std::move(*value);
}

void check(llvm::Error error, const char* what) {
    if (error) throw IRError(std::string(what) + ": " + llvm::toString(std::move(error)));
}

struct HostModule {
    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::Module> module;
    llvm::orc::JITTargetMachineBuilder builder;
    std::string cpu;
};

HostModule emitForHost(const std::vector<loopir::Kernel>& kernels, bool optimize, std::string* emitted) {
    initializeLLVM();
    auto builder = check(llvm::orc::JITTargetMachineBuilder::detectHost(), "detecting the host CPU");
    builder.setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
    auto machine = check(builder.createTargetMachine(), "creating the target machine");
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = emitCpuModule(kernels, *context);
    module->setDataLayout(machine->createDataLayout());
    module->setTargetTriple(machine->getTargetTriple());
    if (emitted) *emitted = moduleText(*module);
    if (optimize) optimizeModule(*module, *machine);
    std::string cpu = machine->getTargetCPU().str();
    return {std::move(context), std::move(module), std::move(builder), cpu};
}

} // namespace

CpuJit::CpuJit(const std::vector<loopir::Kernel>& kernels) {
    HostModule host = emitForHost(kernels, true, &emitted_);
    optimized_ = moduleText(*host.module);
    cpu_ = host.cpu;
    jit_ = check(llvm::orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(host.builder)).create(), "creating the JIT");
    jit_->getMainJITDylib().addGenerator(check(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit_->getDataLayout().getGlobalPrefix()),
        "loading the process's symbols"));
    check(jit_->addIRModule(llvm::orc::ThreadSafeModule(std::move(host.module), std::move(host.context))),
          "adding the module to the JIT");
    for (const auto& k : kernels) {
        auto address = check(jit_->lookup(k.name), "looking up a kernel");
        functions_.push_back(address.toPtr<KernelFn>());
    }
}

CpuJit::~CpuJit() = default;

std::string cpuModuleText(const std::vector<loopir::Kernel>& kernels, bool optimize) {
    HostModule host = emitForHost(kernels, optimize, nullptr);
    return moduleText(*host.module);
}

} // namespace dlc::backends::llvm_emit
