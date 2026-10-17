#pragma once

#include "dlc/passes/Pass.h"

namespace dlc {

// Value numbering over the function's block: an op with the same name, the same operand values, the same result
// type and bit-identical attributes as an earlier op is replaced by that op's result.
class CSEPass : public Pass {
public:
    std::string getName() const override { return "cse"; }
    bool runOnFunction(Function& func) override;
};

} // namespace dlc
