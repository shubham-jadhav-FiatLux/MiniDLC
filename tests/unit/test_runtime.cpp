#include "dlc/runtime/Arena.h"
#include "dlc/runtime/ThreadPool.h"
#include "dlc/runtime/ExecutionPlan.h"
#include <cassert>
#include <iostream>
#include <atomic>

// Declared in test_ir.cpp and test_passes.cpp
int main_ir_tests();
int main_passes_tests();

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

int main() {
    std::cout << "Running MiniDLC Unit Tests...\n";
    main_ir_tests();
    main_passes_tests();
    testArena();
    testThreadPool();
    std::cout << "All Unit Tests Passed Successfully!\n";
    return 0;
}
