#pragma once

#include "dlc/passes/Pass.h"

namespace dlc {

class CSEPass : public Pass {
public:
    std::string getName() const override { return "cse"; }
    bool runOnFunction(Function& func) override;
};

} // namespace dlc
