#include "dlc/runtime/Arena.h"
#include "dlc/runtime/ThreadPool.h"
#include <chrono>
#include <stdexcept>
#include <thread>
#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <vector>

namespace {

using namespace dlc;

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

// A chunk that throws must not leave the others running: they refer to the caller's function and stack. The calling
// thread's chunk throws at once; the call may return only after the slower worker chunks have finished.
TEST(ThreadPool, ParallelChunksWaitsForEveryChunkBeforeRethrowing) {
    runtime::ThreadPool pool(4);
    std::atomic<int> finished{0};
    try {
        pool.parallelChunks(4, [&](int64_t begin, int64_t) {
            if (begin == 3) throw std::runtime_error("chunk 3 failed");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            finished.fetch_add(1);
        });
        FAIL() << "the exception must propagate";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "chunk 3 failed");
        EXPECT_EQ(finished.load(), 3) << "the call returned while worker chunks were still running";
    }
    // A worker's exception propagates too, after the other chunks finish.
    finished = 0;
    EXPECT_THROW(pool.parallelChunks(4, [&](int64_t begin, int64_t) {
        if (begin == 0) throw std::runtime_error("chunk 0 failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        finished.fetch_add(1);
    }), std::runtime_error);
    EXPECT_EQ(finished.load(), 3);
}

} // namespace
