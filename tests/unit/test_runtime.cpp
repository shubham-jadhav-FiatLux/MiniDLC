#include "dlc/runtime/Arena.h"
#include "dlc/runtime/ThreadPool.h"
#include "dlc/runtime/ExecutionPlan.h"
#include "dlc/loopir/LoopIR.h"
#include "dlc/loopir/Lowering.h"
#include "dlc/loopir/LoopPasses.h"
#include "dlc/backends/llvm/LLVMCodeGen.h"
#include "dlc/ir/Builder.h"
#include <cassert>
#include <iostream>
#include <atomic>

// Declared in test_ir.cpp, test_passes.cpp, test_cuda.cpp
int main_ir_tests();
int main_passes_tests();
int main_cuda_tests();

void testArena() {
    dlc::runtime::ArenaAllocator arena(1024 * 1024); // 1 MB
    assert(arena.getCapacity() == 1024 * 1024);
    assert(arena.getRawPointer() != nullptr);

    float* ptr = static_cast<float*>(arena.getOffsetPointer(64));
    *ptr = 42.0f;
    assert(*ptr == 42.0f);
    std::cout << "[PASS] testArena\n";
}

void testThreadPool() {
    dlc::runtime::ThreadPool pool(4);
    std::atomic<int> counter{0};
    pool.parallelFor(0, 100, [&counter](int64_t /*i*/) {
        counter.fetch_add(1, std::memory_order_relaxed);
    });
    assert(counter.load() == 100);
    std::cout << "[PASS] testThreadPool\n";
}

void testLoopIRAndLowering() {

    dlc::IRBuilder builder;
    auto f32_mat = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{128, 256});
    auto func = builder.createFunction("test_lowering", {f32_mat, f32_mat}, {f32_mat});
    auto x = func->getArguments()[0];
    auto y = func->getArguments()[1];

    auto add = builder.createAdd(x, y);
    auto relu = builder.createRelu(add);
    builder.createReturn({relu});

    dlc::FusionPass fusion;
    fusion.runOnFunction(*func);

    auto kernels = dlc::loopir::Lowering::lowerFunction(*func, &fusion, nullptr);
    assert(!kernels.empty());
    assert(kernels[0].getName() == "fused_elementwise_add");

    dlc::loopir::LoopVectorizationPass::run(kernels[0], 8);
    dlc::loopir::LoopParallelizationPass::run(kernels[0]);
    bool bce = dlc::loopir::BoundsCheckEliminationPass::run(kernels[0]);
    assert(bce);
    (void)bce;

    std::string irStr = kernels[0].toString();
    assert(irStr.find("vectorize(8)") != std::string::npos);
    assert(irStr.find("parallel") != std::string::npos);

    std::cout << "[PASS] testLoopIRAndLowering\n";
}

void testLLVMCodeGen() {
    dlc::loopir::KernelNest kernel("test_vec_kernel");
    dlc::loopir::Buffer bOut{"out", dlc::DataType::Float32, {128, 256}, {}, 0, 131072, false, true, false};
    dlc::loopir::Buffer bIn{"in", dlc::DataType::Float32, {128, 256}, {}, 131072, 131072, true, false, false};
    kernel.addBuffer(bIn);
    kernel.addBuffer(bOut);

    auto loop = std::make_shared<dlc::loopir::ForLoop>("idx", 0, 32768);
    loop->setVectorWidth(8);
    kernel.addRootStmt(loop);

    std::string llvmIR = dlc::backends::llvm::LLVMCodeGen::generateKernelIR(kernel);
    assert(llvmIR.find("define void @test_vec_kernel") != std::string::npos);
    assert(llvmIR.find("<8 x float>") != std::string::npos);

    std::cout << "[PASS] testLLVMCodeGen\n";
}

void testLoopVersioningAndBCE() {
    // Non-tile-multiple shape (130 elements with vector width 8: 130 = 16*8 + 2)
    dlc::loopir::KernelNest kernel("test_versioned_kernel");
    dlc::loopir::Buffer bOut{"out", dlc::DataType::Float32, {130}, {}, 0, 130 * 4, false, true, false};
    dlc::loopir::Buffer bIn{"in", dlc::DataType::Float32, {130}, {}, 130 * 4, 130 * 4, true, false, false};
    kernel.addBuffer(bIn);
    kernel.addBuffer(bOut);

    auto loop = std::make_shared<dlc::loopir::ForLoop>("idx", 0, 130);
    loop->setVectorWidth(8);
    auto load = std::make_shared<dlc::loopir::LoadExpr>("in", std::make_shared<dlc::loopir::VarExpr>("idx"));
    auto store = std::make_shared<dlc::loopir::StoreStmt>("out", std::make_shared<dlc::loopir::VarExpr>("idx"), load);
    loop->addBodyStmt(store);
    kernel.addRootStmt(loop);

    bool bce = dlc::loopir::BoundsCheckEliminationPass::run(kernel);
    assert(bce);
    (void)bce;
    assert(kernel.hasBoundsCheckEliminated());

    std::string irStr = kernel.toString();
    assert(irStr.find("0..128 vectorize(8)") != std::string::npos);
    assert(irStr.find("128..130") != std::string::npos);
    assert(irStr.find("attributes {bounds_check_eliminated = true}") != std::string::npos);

    std::cout << "[PASS] testLoopVersioningAndBCE\n";
}

int main() {
    std::cout << "Running MiniDLC Unit Tests...\n";
    main_ir_tests();
    main_passes_tests();
    testArena();
    testThreadPool();
    testLoopIRAndLowering();
    testLLVMCodeGen();
    testLoopVersioningAndBCE();
    main_cuda_tests();
    std::cout << "All Unit Tests Passed Successfully!\n";
    return 0;
}

