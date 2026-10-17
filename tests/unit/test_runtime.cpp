#include "dlc/runtime/Arena.h"
#include "dlc/runtime/ThreadPool.h"
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

} // namespace
