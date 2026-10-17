#pragma once

#include "dlc/passes/Pass.h"
#include <unordered_set>

namespace dlc {

// Which values need adjoints: those that depend on a param argument (varied, a forward dataflow problem) and that
// reach the loss (useful, a backward one), both along differentiable edges only. Integer values, comparisons,
// one-hot encodings, casts and the condition of a select carry no derivative.
class ActivityAnalysis {
public:
    ActivityAnalysis(const Function& func, const Value* loss);

    bool isVaried(const Value* value) const { return varied_.count(value) != 0; }
    bool isUseful(const Value* value) const { return useful_.count(value) != 0; }
    bool isActive(const Value* value) const { return isVaried(value) && isUseful(value); }

    // Whether a derivative flows from the op's result into operand `idx`.
    static bool isDifferentiableOperand(const Operation& op, size_t idx);

private:
    std::unordered_set<const Value*> varied_;
    std::unordered_set<const Value*> useful_;
};

// Marks the ops whose result is active with an `ad.active = 1` annotation (the loss is the first returned value).
class ActivityPass : public Pass {
public:
    std::string getName() const override { return "activity"; }
    bool runOnFunction(Function& func) override;
};

} // namespace dlc
