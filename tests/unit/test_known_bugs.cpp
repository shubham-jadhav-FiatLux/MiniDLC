// Known defects found by the 2026-10-17 audit. Each test states the correct behaviour and stays DISABLED_ until
// the phase that fixes it (see docs/architecture.md); when a phase fixes one, it moves into the regular test files.
// To see the current failures, run: dlc_unit_tests --gtest_also_run_disabled_tests --gtest_filter='KnownBugs.*'
#include "dlc/backends/llvm/LLVMCodeGen.h"
#include "dlc/ir/Builder.h"
#include "dlc/ir/Parser.h"
#include "dlc/loopir/Lowering.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/MemoryPlanning.h"
#include <gtest/gtest.h>
#include <algorithm>

namespace {

using namespace dlc;

bool overlaps(int64_t aOff, int64_t aSize, int64_t bOff, int64_t bSize) {
    return std::max(aOff, bOff) < std::min(aOff + aSize, bOff + bSize);
}

constexpr const char* kMlpLayer = R"(
func @layer(%x: f32[128,784], %w: f32[784,256], %b: f32[256]) -> f32[128,256] {
  %0 = matmul(%x, %w) : f32[128,256]
  %1 = add(%0, %b) : f32[128,256]
  %2 = relu(%1) : f32[128,256]
  return %2
}
)";

// P3: liveness is computed on the unfused schedule, so a fused kernel's output lands inside its own inputs.
TEST(KnownBugs, DISABLED_FusedKernelOutputDoesNotAliasItsInputs) {
    auto mod = IRParser::parseModule(kMlpLayer);
    auto& func = *mod->getFunctions()[0];
    FusionPass fusion;
    fusion.runOnFunction(func);
    MemoryPlanningPass memPlan;
    memPlan.runOnFunction(func);

    for (const auto& kernel : loopir::Lowering::lowerFunction(func, &fusion, &memPlan)) {
        for (const auto& out : kernel.getBuffers()) {
            if (!out.isOutput) continue;
            for (const auto& in : kernel.getBuffers()) {
                if (in.isOutput) continue;
                EXPECT_FALSE(overlaps(out.arenaOffsetBytes, out.sizeBytes, in.arenaOffsetBytes, in.sizeBytes))
                    << kernel.getName() << ": output %" << out.name << " overlaps input %" << in.name;
            }
        }
    }
}

// P3: parameters persist across training steps, so no other value may ever be placed on top of them.
TEST(KnownBugs, DISABLED_ParameterBuffersAreNeverReused) {
    auto mod = IRParser::parseModule(kMlpLayer);
    auto& func = *mod->getFunctions()[0];
    MemoryPlanningPass memPlan;
    memPlan.runOnFunction(func);

    const auto& offsets = memPlan.getOffsets();
    for (const auto& arg : func.getArguments()) {
        const auto& a = offsets.at(arg.get());
        for (const auto& op : func.getEntryBlock()->getOperations()) {
            for (const auto& result : op->getResults()) {
                const auto& r = offsets.at(result.get());
                EXPECT_FALSE(overlaps(a.offsetBytes, a.sizeBytes, r.offsetBytes, r.sizeBytes))
                    << "parameter " << arg->toString() << " shares memory with " << op->getOpName();
            }
        }
    }
}

// P4: the LLVM emitter must translate the loop IR; v1 picks a template by kernel name and turns div into fadd.
TEST(KnownBugs, DISABLED_LLVMCodegenLowersDivision) {
    auto mod = IRParser::parseModule(R"(
func @d(%a: f32[64], %b: f32[64]) -> f32[64] {
  %0 = div(%a, %b) : f32[64]
  return %0
}
)");
    auto& func = *mod->getFunctions()[0];
    FusionPass fusion;
    fusion.runOnFunction(func);
    auto kernels = loopir::Lowering::lowerFunction(func, &fusion, nullptr);
    ASSERT_EQ(kernels.size(), 1u);
    EXPECT_NE(backends::llvm::LLVMCodeGen::generateKernelIR(kernels[0]).find("fdiv"), std::string::npos);
}

} // namespace
