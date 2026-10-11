#pragma once

#include "dlc/passes/Pass.h"
#include <vector>

namespace dlc {

class AutodiffPass : public Pass {
public:
    AutodiffPass() = default;
    explicit AutodiffPass(std::vector<std::string> paramNames)
        : paramNames_(std::move(paramNames)) {}

    std::string getName() const override { return "autodiff"; }
    bool runOnFunction(Function& func) override;

private:
    std::vector<std::string> paramNames_;
};

} // namespace dlc
