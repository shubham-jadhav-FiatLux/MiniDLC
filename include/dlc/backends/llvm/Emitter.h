#pragma once

#include "dlc/loopir/LoopIR.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
} // namespace llvm

namespace dlc::backends::llvm_emit {

// Translates loop kernels into LLVM IR. On the CPU each kernel is
//   void @name(ptr %args, i64 %begin, i64 %end)
// whose `args` array holds the global buffers in order; the body sits in an always-inline function that takes the
// buffers as `noalias` pointers (buffers of one launch never overlap: values that share storage share a buffer). The
// outermost Parallel loop runs [begin, end). Unrolled loops are expanded here, so register tiles become SSA values;
// Vectorized loops carry their width as llvm.loop metadata and allow reassociation of their reductions; every other
// decision is the schedule's.
std::unique_ptr<llvm::Module> emitCpuModule(const std::vector<loopir::Kernel>& kernels, llvm::LLVMContext& context);

// Runs LLVM's O3 pipeline for `machine` (its TTI drives the vectorizer's cost model).
void optimizeModule(llvm::Module& module, llvm::TargetMachine& machine);

std::string moduleText(const llvm::Module& module);

} // namespace dlc::backends::llvm_emit
