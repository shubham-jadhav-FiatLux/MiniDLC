#pragma once

#include "dlc/passes/Pass.h"

namespace dlc {

struct OptimizerConfig {
    enum class Kind { SGD, Adam };
    Kind kind{Kind::Adam};
    double lr{1e-3};
    double momentum{0.0};                          // SGD
    double beta1{0.9}, beta2{0.999}, eps{1e-8};   // Adam
};

// Turns the gradients that autodiff returned into a parameter update, so one function is one training step. Each
// `{grad = %p}` result is replaced by the new value of %p, returned as `{donates = %p}` (it takes over %p's
// buffer). The optimizer's state becomes state arguments with an initial value: per parameter a velocity (SGD with
// momentum) or the moments %p.m and %p.v (Adam), and for Adam the bias-correction powers β1^t and β2^t.
class OptimizerPass : public Pass {
public:
    explicit OptimizerPass(OptimizerConfig config) : config_(config) {}

    std::string getName() const override { return config_.kind == OptimizerConfig::Kind::Adam ? "adam" : "sgd"; }
    bool runOnFunction(Function& func) override;

private:
    OptimizerConfig config_;
};

} // namespace dlc
