#include "dlc/backends/llvm/Ptx.h"
#include "dlc/backends/llvm/Emitter.h"
#include "dlc/ir/Error.h"
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>
#include <memory>
#include <mutex>

extern "C" {
void LLVMInitializeNVPTXTargetInfo();
void LLVMInitializeNVPTXTarget();
void LLVMInitializeNVPTXTargetMC();
void LLVMInitializeNVPTXAsmPrinter();
}

namespace dlc::backends::llvm_emit {

PtxModule compilePtx(const std::vector<loopir::Kernel>& kernels, const std::string& arch) {
    static std::once_flag once;
    std::call_once(once, [] {
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();
    });
    llvm::Triple triple("nvptx64-nvidia-cuda");
    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) throw IRError("PTX: " + error);
    std::unique_ptr<llvm::TargetMachine> machine(
        target->createTargetMachine(triple, arch, "", llvm::TargetOptions(), std::nullopt, std::nullopt,
                                    llvm::CodeGenOptLevel::Aggressive));
    if (!machine) throw IRError("PTX: no target machine for " + arch);

    llvm::LLVMContext context;
    auto module = emitGpuModule(kernels, context, *machine);
    PtxModule out;
    out.arch = arch;
    out.emittedIR = moduleText(*module);
    optimizeModule(*module, *machine);
    out.optimizedIR = moduleText(*module);

    llvm::SmallString<0> ptx;
    llvm::raw_svector_ostream os(ptx);
    llvm::legacy::PassManager pm;
    if (machine->addPassesToEmitFile(pm, os, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
        throw IRError("PTX: the NVPTX target cannot emit assembly");
    }
    pm.run(*module);
    out.ptx = ptx.str().str();
    return out;
}

} // namespace dlc::backends::llvm_emit
