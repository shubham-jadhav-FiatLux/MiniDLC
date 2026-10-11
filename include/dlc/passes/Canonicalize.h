#pragma once

#include "dlc/passes/Pass.h"

namespace dlc {

class CanonicalizePass : public Pass {
public:
    std::string getName() const override { return "canonicalize"; }
    bool runOnFunction(Function& func) override;
};

} // namespace dlc
