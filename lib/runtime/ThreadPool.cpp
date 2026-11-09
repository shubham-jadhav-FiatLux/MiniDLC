#include "dlc/runtime/ThreadPool.h"
#include <exception>

namespace dlc::runtime {

ThreadPool::ThreadPool(size_t threads) {
    if (threads == 0) threads = 1;
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex_);
                    this->cv_.wait(lock, [this]() {
                        return this->stop_ || !this->tasks_.empty();
                    });
                    if (this->stop_ && this->tasks_.empty()) {
                        return;
                    }
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

namespace {

// Waits for every future, then rethrows the first exception (if any) — after `first`, the caller's own. A task must
// not outlive the call: the tasks refer to the caller's function and stack.
void waitAll(std::vector<std::future<void>>& futures, std::exception_ptr first = nullptr) {
    for (auto& f : futures) {
        try {
            f.get();
        } catch (...) {
            if (!first) first = std::current_exception();
        }
    }
    if (first) std::rethrow_exception(first);
}

} // namespace

void ThreadPool::parallelFor(int64_t start, int64_t end, const std::function<void(int64_t)>& func) {
    if (start >= end) return;
    int64_t total = end - start;
    size_t numWorkers = workers_.size();
    if (numWorkers <= 1 || total <= 1) {
        for (int64_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }

    int64_t chunkSize = (total + numWorkers - 1) / numWorkers;
    std::vector<std::future<void>> futures;

    for (size_t w = 0; w < numWorkers; ++w) {
        int64_t chunkStart = start + w * chunkSize;
        int64_t chunkEnd = std::min(end, chunkStart + chunkSize);
        if (chunkStart >= end) break;

        futures.push_back(enqueue([chunkStart, chunkEnd, &func]() {
            for (int64_t i = chunkStart; i < chunkEnd; ++i) {
                func(i);
            }
        }));
    }
    waitAll(futures);
}

void ThreadPool::parallelChunks(int64_t extent, const std::function<void(int64_t, int64_t)>& func) {
    if (extent <= 0) return;
    int64_t workers = static_cast<int64_t>(workers_.size());
    if (workers <= 1 || extent == 1) {
        func(0, extent);
        return;
    }
    int64_t chunks = std::min(workers, extent);
    int64_t base = extent / chunks;
    int64_t extra = extent % chunks;
    std::vector<std::future<void>> futures;
    std::exception_ptr own;
    int64_t begin = 0;
    for (int64_t c = 0; c < chunks; ++c) {
        int64_t end = begin + base + (c < extra ? 1 : 0);
        if (c == chunks - 1) {
            try {
                func(begin, end); // the calling thread takes the last chunk
            } catch (...) {
                own = std::current_exception();
            }
        } else {
            futures.push_back(enqueue([begin, end, &func]() { func(begin, end); }));
        }
        begin = end;
    }
    waitAll(futures, own);
}

} // namespace dlc::runtime
