#include "dlc/passes/Optimizer.h"
#include "dlc/ir/Builder.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/SlotTracker.h"
#include <set>

namespace dlc {

bool OptimizerPass::runOnFunction(Function& func) {
    Block* entry = func.getEntryBlock();
    const auto& ops = entry->getOperations();
    if (ops.empty() || ops.back()->getOpName() != "return") throw IRError("optimizer: the function has no 'return'");
    Operation* ret = ops.back().get();

    std::vector<ValuePtr> outputs;
    std::vector<TypePtr> types;
    std::vector<ResultInfo> info;
    std::vector<std::pair<size_t, ValuePtr>> grads; // (param argument, gradient)
    const auto& resultInfo = func.getResultInfo();
    for (size_t i = 0; i < ret->getNumOperands(); ++i) {
        const ValuePtr& value = ret->getOperand(i);
        switch (resultInfo.at(i).kind) {
            case ResultKind::Grad: grads.emplace_back(resultInfo[i].arg, value); break;
            case ResultKind::Value:
                outputs.push_back(value);
                types.push_back(value->getType());
                info.emplace_back();
                break;
            case ResultKind::Donates: throw IRError("optimizer: the function already updates its parameters");
        }
    }
    if (grads.empty()) throw IRError("optimizer: the function returns no gradients; run autodiff first");

    // State arguments are named after their parameter, e.g. %fc1.weight.m.
    SlotTracker names(func);
    std::set<std::string> taken;
    for (const auto& arg : func.getArguments()) taken.insert(names.name(arg.get()));
    auto addState = [&](const TypePtr& type, std::string name, double init) {
        std::string unique = name;
        for (int k = 1; taken.count(unique); ++k) unique = name + "_" + std::to_string(k);
        taken.insert(unique);
        func.addArgument(type, unique, ArgRole::State, init);
        return func.getArguments().back();
    };
    auto donate = [&](const ValuePtr& value, const ValuePtr& arg) {
        outputs.push_back(value);
        types.push_back(value->getType());
        info.push_back({ResultKind::Donates, static_cast<size_t>(func.getArgIndex(arg.get()))});
    };

    entry->eraseOperation(ret);
    IRBuilder b(entry);
    auto scalar = [&b](double v) { return b.createScalar(v); };
    const OptimizerConfig& c = config_;

    if (c.kind == OptimizerConfig::Kind::SGD) {
        for (const auto& [argIdx, g] : grads) {
            ValuePtr theta = func.getArguments()[argIdx];
            std::string name = names.name(theta.get());
            if (c.momentum > 0.0) {
                // v ← μ v + g;  θ ← θ - lr v
                ValuePtr velocity = addState(theta->getType(), name + ".velocity", 0.0);
                ValuePtr v = b.createAdd(b.createMul(velocity, scalar(c.momentum)), g);
                donate(b.createSub(theta, b.createMul(v, scalar(c.lr))), theta);
                donate(v, velocity);
            } else {
                donate(b.createSub(theta, b.createMul(g, scalar(c.lr))), theta);
            }
        }
    } else {
        // m ← β1 m + (1-β1) g;  v ← β2 v + (1-β2) g²;  m̂ = m / (1-β1^t);  v̂ = v / (1-β2^t);
        // θ ← θ - lr m̂ / (√v̂ + ε). The powers β^t are state too, starting at 1.
        auto scalarType = std::make_shared<TensorType>(DataType::Float32, Shape{});
        ValuePtr beta1Pow = addState(scalarType, "adam.beta1_pow", 1.0);
        ValuePtr beta2Pow = addState(scalarType, "adam.beta2_pow", 1.0);
        ValuePtr beta1PowNext = b.createMul(beta1Pow, scalar(c.beta1));
        ValuePtr beta2PowNext = b.createMul(beta2Pow, scalar(c.beta2));
        ValuePtr correction1 = b.createSub(scalar(1.0), beta1PowNext);
        ValuePtr correction2 = b.createSub(scalar(1.0), beta2PowNext);
        for (const auto& [argIdx, g] : grads) {
            ValuePtr theta = func.getArguments()[argIdx];
            std::string name = names.name(theta.get());
            ValuePtr m = addState(theta->getType(), name + ".m", 0.0);
            ValuePtr v = addState(theta->getType(), name + ".v", 0.0);
            // One op per statement: argument evaluation order is unspecified, and the op order must not depend on
            // the compiler.
            ValuePtr mDecay = b.createMul(m, scalar(c.beta1));
            ValuePtr mNew = b.createMul(g, scalar(1.0 - c.beta1));
            ValuePtr mNext = b.createAdd(mDecay, mNew);
            ValuePtr vDecay = b.createMul(v, scalar(c.beta2));
            ValuePtr vNew = b.createMul(b.createMul(g, g), scalar(1.0 - c.beta2));
            ValuePtr vNext = b.createAdd(vDecay, vNew);
            ValuePtr mHat = b.createDiv(mNext, correction1);
            ValuePtr vHat = b.createDiv(vNext, correction2);
            ValuePtr scaled = b.createMul(mHat, scalar(c.lr));
            ValuePtr denom = b.createAdd(b.createUnary("sqrt", vHat), scalar(c.eps));
            ValuePtr step = b.createDiv(scaled, denom);
            donate(b.createSub(theta, step), theta);
            donate(mNext, m);
            donate(vNext, v);
        }
        donate(beta1PowNext, beta1Pow);
        donate(beta2PowNext, beta2Pow);
    }

    b.createReturn(outputs);
    func.setResults(types, info);
    return true;
}

} // namespace dlc
