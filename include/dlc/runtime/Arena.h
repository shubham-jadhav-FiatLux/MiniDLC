#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

namespace dlc::runtime {

class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacityBytes);
    ~ArenaAllocator();

    void* getRawPointer() { return data_; }
    const void* getRawPointer() const { return data_; }
    size_t getCapacity() const { return capacityBytes_; }

    void* getOffsetPointer(int64_t offsetBytes) {
        return static_cast<char*>(data_) + offsetBytes;
    }

    const void* getOffsetPointer(int64_t offsetBytes) const {
        return static_cast<const char*>(data_) + offsetBytes;
    }

private:
    size_t capacityBytes_{0};
    void* data_{nullptr};
};

} // namespace dlc::runtime
