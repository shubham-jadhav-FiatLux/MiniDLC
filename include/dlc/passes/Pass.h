#pragma once

#include "dlc/ir/Module.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace dlc {

class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string getName() const = 0;
    virtual bool runOnModule(Module& module);
    virtual bool runOnFunction(Function& func);
};

using PassPtr = std::unique_ptr<Pass>;

class PassManager {
public:
    PassManager() = default;

    void addPass(PassPtr pass) { passes_.push_back(std::move(pass)); }
    void setPrintAfterAll(bool enable) { printAfterAll_ = enable; }

    bool run(Module& module);

private:
    std::vector<PassPtr> passes_;
    bool printAfterAll_{false};
};

} // namespace dlc
