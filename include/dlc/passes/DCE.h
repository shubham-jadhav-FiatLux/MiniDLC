#pragma once

#include "dlc/passes/Pass.h"

namespace dlc {

class DCEPass : public Pass {
public:
    std::string getName() const override { return "dce"; }
    bool runOnFunction(Function& func) override;
};

} // namespace dlc
