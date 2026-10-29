#include "dlc/runtime/Executable.h"
#include "dlc/ir/Error.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace dlc::runtime {

namespace {

constexpr size_t kAlignment = 256;

char* allocate(int64_t bytes) {
    size_t size = std::max<size_t>(kAlignment, (static_cast<size_t>(bytes) + kAlignment - 1) / kAlignment * kAlignment);
    void* p = std::aligned_alloc(kAlignment, size);
    if (!p) throw IRError("runtime: out of memory allocating " + std::to_string(bytes) + " bytes");
    std::memset(p, 0, size);
    return static_cast<char*>(p);
}

void fill(char* p, const PlanArg& a, double value) {
    int64_t n = a.slot.bytes / 4;
    if (a.dtype == DataType::Float32) {
        float v = static_cast<float>(value);
        for (int64_t i = 0; i < n; ++i) std::memcpy(p + 4 * i, &v, 4);
    } else {
        int32_t v = static_cast<int32_t>(value);
        for (int64_t i = 0; i < n; ++i) std::memcpy(p + 4 * i, &v, 4);
    }
}

} // namespace

Executable::Executable(Plan plan, std::unique_ptr<KernelRunner> runner, size_t threads)
    : plan_(std::move(plan)), runner_(std::move(runner)) {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    pool_ = std::make_unique<ThreadPool>(threads);
    persistent_ = allocate(plan_.persistentBytes);
    transient_ = allocate(plan_.transientBytes);
    resetState();
    for (const Launch& l : plan_.launches) {
        bool seen = std::any_of(profile_.begin(), profile_.end(), [&](const KernelProfile& p) { return p.name == l.name; });
        if (!seen) profile_.push_back({l.name, 0, 0.0, l.flops, l.bytes});
    }
}

Executable::~Executable() {
    std::free(persistent_);
    std::free(transient_);
}

char* Executable::address(const Slot& slot) const {
    return (slot.region == Region::Persistent ? persistent_ : transient_) + slot.offset;
}

const PlanArg& Executable::arg(const std::string& name) const {
    for (const auto& a : plan_.args) {
        if (a.name == name) return a;
    }
    throw IRError("runtime: no argument named '" + name + "'");
}

void Executable::resetState() {
    for (const auto& a : plan_.args) {
        if (a.role == ArgRole::State) fill(address(a.slot), a, a.init.value_or(0.0));
    }
}

void Executable::setArg(const std::string& name, const void* data, size_t bytes) {
    const PlanArg& a = arg(name);
    if (static_cast<int64_t>(bytes) != a.slot.bytes) {
        throw IRError("runtime: '" + name + "' takes " + std::to_string(a.slot.bytes) + " bytes, got " +
                      std::to_string(bytes));
    }
    std::memcpy(address(a.slot), data, bytes);
}

HostTensor Executable::getArg(const std::string& name) const {
    const PlanArg& a = arg(name);
    HostTensor t{a.dtype, a.shape, std::vector<char>(a.slot.bytes)};
    std::memcpy(t.data.data(), address(a.slot), a.slot.bytes);
    return t;
}

std::vector<HostTensor> Executable::run(const std::unordered_map<std::string, TensorView>& inputs) {
    for (const auto& a : plan_.args) {
        if (a.role != ArgRole::Input) continue;
        auto it = inputs.find(a.name);
        if (it == inputs.end()) throw IRError("runtime: missing input '" + a.name + "'");
        const TensorView& t = it->second;
        if (t.dtype != a.dtype || t.shape != a.shape || static_cast<int64_t>(t.bytes) != a.slot.bytes) {
            throw IRError("runtime: input '" + a.name + "' has the wrong type or shape");
        }
        std::memcpy(address(a.slot), t.data, t.bytes);
    }
    std::vector<void*> args;
    for (const Launch& l : plan_.launches) {
        args.clear();
        for (const Slot& s : l.args) args.push_back(address(s));
        auto start = std::chrono::steady_clock::now();
        if (l.parallelExtent > 1) {
            pool_->parallelChunks(l.parallelExtent, [&](int64_t b, int64_t e) { runner_->run(l.kernel, args.data(), b, e); });
        } else {
            runner_->run(l.kernel, args.data(), 0, 1);
        }
        if (profiling_) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            for (auto& p : profile_) {
                if (p.name == l.name) {
                    ++p.calls;
                    p.seconds += seconds;
                }
            }
        }
    }
    for (const Copy& c : plan_.copies) std::memmove(address(c.to), address(c.from), c.from.bytes);
    std::vector<HostTensor> results;
    for (const auto& r : plan_.results) {
        if (r.kind == ResultKind::Donates) continue;
        HostTensor t{r.dtype, r.shape, std::vector<char>(r.slot.bytes)};
        std::memcpy(t.data.data(), address(r.slot), r.slot.bytes);
        results.push_back(std::move(t));
    }
    return results;
}

std::vector<KernelProfile> Executable::profile() const { return profile_; }

void Executable::clearProfile() {
    for (auto& p : profile_) {
        p.calls = 0;
        p.seconds = 0.0;
    }
}

} // namespace dlc::runtime
