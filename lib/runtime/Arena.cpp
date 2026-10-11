#include "dlc/runtime/Arena.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace dlc::runtime {

ArenaAllocator::ArenaAllocator(size_t capacityBytes) : capacityBytes_(capacityBytes) {
    if (capacityBytes_ > 0) {
        // 64-byte aligned allocation
        if (posix_memalign(&data_, 64, capacityBytes_) != 0) {
            throw std::runtime_error("Failed to allocate aligned arena memory");
        }
        std::memset(data_, 0, capacityBytes_);
    }
}

ArenaAllocator::~ArenaAllocator() {
    if (data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

} // namespace dlc::runtime
