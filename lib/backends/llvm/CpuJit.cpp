#include "dlc/backends/llvm/CpuJit.h"
#include "dlc/backends/llvm/Emitter.h"
#include "dlc/backends/llvm/KernelCache.h"
#include "dlc/ir/Error.h"
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
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

CpuObject compileCpuObject(const std::vector<loopir::Kernel>& kernels, bool keepIR) {
    initializeLLVM();
    auto builder = check(llvm::orc::JITTargetMachineBuilder::detectHost(), "detecting the host CPU");
    builder.setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
    auto machine = check(builder.createTargetMachine(), "creating the target machine");
    CpuObject out;
    out.triple = machine->getTargetTriple().str();
    out.cpu = machine->getTargetCPU().str();
    out.features = machine->getTargetFeatureString().str();

    // The cache key: the kernels and everything about the target that changes the code.
    // (This file's build time too: it decides how the object file is produced.)
    std::string content = out.triple + "\n" + out.cpu + "\n" + out.features + "\n" __DATE__ " " __TIME__ "\n";
    for (const auto& k : kernels) content += loopir::toString(k);
    KernelCache& cache = KernelCache::instance();
    const std::string key = cache.key("cpu", content);
    if (!keepIR) {
        if (auto cached = cache.load(key)) {
            out.object = std::move(*cached);
            out.fromCache = true;
            return out;
        }
    }
    HostModule host = emitForHost(kernels, true, keepIR ? &out.emittedIR : nullptr);
    if (keepIR) out.optimizedIR = moduleText(*host.module);
    llvm::SmallString<0> buffer;
    llvm::raw_svector_ostream os(buffer);
    llvm::legacy::PassManager pm;
    if (machine->addPassesToEmitFile(pm, os, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        throw IRError("CPU JIT: the target cannot emit object files");
    }
    pm.run(*host.module);
    out.object = buffer.str().str();
    cache.store(key, out.object);
    return out;
}

std::string hostIncompatibility(const std::string& triple, const std::string& features) {
    initializeLLVM();
    auto builder = check(llvm::orc::JITTargetMachineBuilder::detectHost(), "detecting the host CPU");
    if (builder.getTargetTriple().str() != triple) {
        return "compiled for " + triple + ", the host is " + builder.getTargetTriple().str();
    }
    llvm::StringMap<bool> host = llvm::sys::getHostCPUFeatures();
    std::string missing;
    llvm::SmallVector<llvm::StringRef> wanted;
    llvm::StringRef(features).split(wanted, ',', -1, false);
    for (llvm::StringRef f : wanted) {
        if (!f.consume_front("+")) continue;
        auto it = host.find(f);
        if (it == host.end() || !it->second) missing += (missing.empty() ? "" : ", ") + f.str();
    }
    return missing.empty() ? "" : "the host CPU lacks " + missing;
}

CpuJit::CpuJit(const std::vector<loopir::Kernel>& kernels, bool keepIR) {
    CpuObject compiled = compileCpuObject(kernels, keepIR);
    emitted_ = std::move(compiled.emittedIR);
    optimized_ = std::move(compiled.optimizedIR);
    cpu_ = compiled.cpu;
    fromCache_ = compiled.fromCache;
    std::vector<std::string> names;
    for (const auto& k : kernels) names.push_back(k.name);
    link(compiled.object, names);
}

CpuJit::CpuJit(const std::string& object, const std::vector<std::string>& names) {
    initializeLLVM();
    link(object, names);
}

void CpuJit::link(const std::string& object, const std::vector<std::string>& names) {
    auto builder = check(llvm::orc::JITTargetMachineBuilder::detectHost(), "detecting the host CPU");
    builder.setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
    jit_ = check(llvm::orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(builder)).create(), "creating the JIT");
    jit_->getMainJITDylib().addGenerator(check(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit_->getDataLayout().getGlobalPrefix()),
        "loading the process's symbols"));
    check(jit_->addObjectFile(llvm::MemoryBuffer::getMemBufferCopy(object, "minidlc-kernels")), "adding the object file to the JIT");
    for (const auto& name : names) {
        auto address = check(jit_->lookup(name), "looking up a kernel");
        functions_.push_back(address.toPtr<KernelFn>());
    }
}

CpuJit::~CpuJit() = default;

std::string cpuModuleText(const std::vector<loopir::Kernel>& kernels, bool optimize) {
    HostModule host = emitForHost(kernels, optimize, nullptr);
    return moduleText(*host.module);
}

} // namespace dlc::backends::llvm_emit
