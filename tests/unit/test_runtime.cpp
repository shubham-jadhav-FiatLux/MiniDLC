#include "dlc/backends/llvm/LLVMCodeGen.h"
#include "dlc/ir/Builder.h"
#include "dlc/loopir/LoopIR.h"
#include "dlc/loopir/LoopPasses.h"
#include "dlc/loopir/Lowering.h"
#include "dlc/runtime/Arena.h"
#include "dlc/runtime/ThreadPool.h"
#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <vector>

namespace {

using namespace dlc;

TypePtr f32(Shape shape) { return std::make_shared<TensorType>(DataType::Float32, std::move(shape)); }

TEST(Arena, AllocatesAlignedZeroedMemory) {
    runtime::ArenaAllocator arena(1024 * 1024);
    EXPECT_EQ(arena.getCapacity(), 1024u * 1024u);
    ASSERT_NE(arena.getRawPointer(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(arena.getRawPointer()) % 64, 0u);

    float* ptr = static_cast<float*>(arena.getOffsetPointer(64));
    EXPECT_EQ(*ptr, 0.0f);
    *ptr = 42.0f;
    EXPECT_EQ(*static_cast<float*>(arena.getOffsetPointer(64)), 42.0f);
}

TEST(ThreadPool, ParallelForVisitsEveryIndexOnce) {
    runtime::ThreadPool pool(4);
    std::vector<std::atomic<int>> visits(100);
    pool.parallelFor(0, 100, [&visits](int64_t i) { visits[i].fetch_add(1, std::memory_order_relaxed); });
    for (size_t i = 0; i < visits.size(); ++i) EXPECT_EQ(visits[i].load(), 1) << "index " << i;
}

TEST(LoopIR, LowersFusedAddReluToOneVectorizedParallelLoop) {
    IRBuilder builder;
    auto func = builder.createFunction("test_lowering", {f32({128, 256}), f32({128, 256})}, {f32({128, 256})});
    const auto& args = func->getArguments();
    auto add = builder.createAdd(args[0], args[1]);
    auto relu = builder.createRelu(add);
    builder.createReturn({relu});

    FusionPass fusion;
    fusion.runOnFunction(*func);
    auto kernels = loopir::Lowering::lowerFunction(*func, &fusion, nullptr);
    ASSERT_EQ(kernels.size(), 1u);
    EXPECT_EQ(kernels[0].getName(), "fused_elementwise_add");

    loopir::LoopVectorizationPass::run(kernels[0], 8);
    loopir::LoopParallelizationPass::run(kernels[0]);
    EXPECT_TRUE(loopir::BoundsCheckEliminationPass::run(kernels[0]));

    std::string irStr = kernels[0].toString();
    EXPECT_NE(irStr.find("parallel for idx in 0..32768 vectorize(8)"), std::string::npos) << irStr;
    EXPECT_NE(irStr.find("max((arg0[idx] + arg1[idx]), 0.0)"), std::string::npos) << irStr;
}

// Text only: the v1 emitter picks a template by kernel name; it is not compiled or executed here.
TEST(LLVMCodeGen, EmitsVectorKernelText) {
    loopir::KernelNest kernel("test_vec_kernel");
    loopir::Buffer bOut{"out", DataType::Float32, {128, 256}, {}, 0, 131072, false, true, false};
    loopir::Buffer bIn{"in", DataType::Float32, {128, 256}, {}, 131072, 131072, true, false, false};
    kernel.addBuffer(bIn);
    kernel.addBuffer(bOut);

    auto loop = std::make_shared<loopir::ForLoop>("idx", 0, 32768);
    loop->setVectorWidth(8);
    kernel.addRootStmt(loop);

    std::string llvmIR = backends::llvm::LLVMCodeGen::generateKernelIR(kernel);
    EXPECT_NE(llvmIR.find("define void @test_vec_kernel"), std::string::npos);
    EXPECT_NE(llvmIR.find("<8 x float>"), std::string::npos);
}

TEST(BoundsCheckElimination, PeelsNonMultipleVectorLoopIntoMainAndTail) {
    // 130 elements with vector width 8: 130 = 16 * 8 + 2
    loopir::KernelNest kernel("test_versioned_kernel");
    loopir::Buffer bOut{"out", DataType::Float32, {130}, {}, 0, 130 * 4, false, true, false};
    loopir::Buffer bIn{"in", DataType::Float32, {130}, {}, 130 * 4, 130 * 4, true, false, false};
    kernel.addBuffer(bIn);
    kernel.addBuffer(bOut);

    auto loop = std::make_shared<loopir::ForLoop>("idx", 0, 130);
    loop->setVectorWidth(8);
    auto load = std::make_shared<loopir::LoadExpr>("in", std::make_shared<loopir::VarExpr>("idx"));
    loop->addBodyStmt(std::make_shared<loopir::StoreStmt>("out", std::make_shared<loopir::VarExpr>("idx"), load));
    kernel.addRootStmt(loop);

    EXPECT_TRUE(loopir::BoundsCheckEliminationPass::run(kernel));
    EXPECT_TRUE(kernel.hasBoundsCheckEliminated());

    std::string irStr = kernel.toString();
    EXPECT_NE(irStr.find("for idx in 0..128 vectorize(8)"), std::string::npos) << irStr;
    EXPECT_NE(irStr.find("for idx in 128..130 {"), std::string::npos) << irStr;
    EXPECT_NE(irStr.find("attributes {bounds_check_eliminated = true}"), std::string::npos) << irStr;
}

TEST(BoundsCheckElimination, RejectsOutOfBoundsAccess) {
    loopir::KernelNest kernel("test_oob_kernel");
    loopir::Buffer bOut{"out", DataType::Float32, {64}, {}, 0, 64 * 4, false, true, false};
    loopir::Buffer bIn{"in", DataType::Float32, {32}, {}, 64 * 4, 32 * 4, true, false, false};
    kernel.addBuffer(bIn);
    kernel.addBuffer(bOut);

    auto loop = std::make_shared<loopir::ForLoop>("idx", 0, 64);
    auto load = std::make_shared<loopir::LoadExpr>("in", std::make_shared<loopir::VarExpr>("idx"));
    loop->addBodyStmt(std::make_shared<loopir::StoreStmt>("out", std::make_shared<loopir::VarExpr>("idx"), load));
    kernel.addRootStmt(loop);

    EXPECT_FALSE(loopir::BoundsCheckEliminationPass::run(kernel));
    EXPECT_FALSE(kernel.hasBoundsCheckEliminated());
}

} // namespace
