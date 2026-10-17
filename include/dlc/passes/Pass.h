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
    // Verify the module after every pass; a failure raises IRError naming the pass.
    void setVerifyEach(bool enable) { verifyEach_ = enable; }

    // Errors raised by a pass are rethrown as IRError prefixed with the pass name.
    bool run(Module& module);

private:
    std::vector<PassPtr> passes_;
    bool printAfterAll_{false};
    bool verifyEach_{false};
};

} // namespace dlc
