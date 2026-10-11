#include "dlc/runtime/ThreadPool.h"

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

    for (auto& fut : futures) {
        fut.get();
    }
}

} // namespace dlc::runtime
