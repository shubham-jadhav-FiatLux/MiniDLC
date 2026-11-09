#pragma once

#include "dlc/runtime/Plan.h"
#include "dlc/runtime/ThreadPool.h"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dlc::runtime {

// Runs the kernels of a plan. `args` holds the address of each global buffer of the kernel; on the CPU, [begin, end)
// is the share of the outermost parallel loop for this call (0, 1 when there is none).
class KernelRunner {
public:
    virtual ~KernelRunner() = default;
    // Around each launch, before its first and after its last call to run() (which may come from several threads).
    virtual void beginLaunch(int /*kernel*/) {}
    virtual void endLaunch(int /*kernel*/) {}
    virtual void run(int kernel, void** args, int64_t begin, int64_t end) = 0;
    virtual std::string name() const = 0;
};

// A host-side tensor: element type, shape and the bytes of a row-major array.
struct HostTensor {
    DataType dtype{DataType::Float32};
    Shape shape;
    std::vector<char> data;
};

// A borrowed row-major array.
struct TensorView {
    DataType dtype{DataType::Float32};
    Shape shape;
    const void* data{nullptr};
    size_t bytes{0};
};

struct KernelProfile {
    std::string name;
    int64_t calls{0};
    double seconds{0.0};
    double flops{0.0};
    int64_t bytes{0};
};

// A compiled step function with its memory: the persistent region holds params, state and consts across calls; the
// transient arena is reused every call. run() copies the inputs in, runs the launches, applies the final copies and
// returns the results that are not donated.
class Executable {
public:
    Executable(Plan plan, std::unique_ptr<KernelRunner> runner, size_t threads = 0);
    ~Executable();

    const Plan& plan() const { return plan_; }
    std::vector<HostTensor> run(const std::unordered_map<std::string, TensorView>& inputs);

    // Params, state and consts by name.
    void setArg(const std::string& name, const void* data, size_t bytes);
    HostTensor getArg(const std::string& name) const;
    // Resets every state argument to its initial value.
    void resetState();

    void setProfiling(bool enabled) { profiling_ = enabled; }
    std::vector<KernelProfile> profile() const;
    void clearProfile();

private:
    char* address(const Slot& slot) const;
    const PlanArg& arg(const std::string& name) const;

    Plan plan_;
    std::unique_ptr<KernelRunner> runner_;
    std::unique_ptr<ThreadPool> pool_;
    char* persistent_{nullptr};
    char* transient_{nullptr};
    bool profiling_{false};
    std::vector<KernelProfile> profile_;
};

} // namespace dlc::runtime
