#include "dlc/passes/Activity.h"

namespace dlc {

bool ActivityAnalysis::isDifferentiableOperand(const Operation& op, size_t idx) {
    const ValuePtr& operand = op.getOperand(idx);
    if (!operand || !operand->getType() || !operand->getType()->isTensor() ||
        static_cast<const TensorType&>(*operand->getType()).getElementType() != DataType::Float32) {
        return false;
    }
    const std::string& name = op.getOpName();
    if (name == "cmp_gt" || name == "onehot" || name == "cast" || name == "return") return false;
    if (name == "select") return idx != 0;
    if (name == "scatter_add") return idx == 1;
    if (name == "embedding" || name == "cross_entropy") return idx == 0;
    return true;
}

ActivityAnalysis::ActivityAnalysis(const Function& func, const Value* loss) {
    const auto& args = func.getArguments();
    for (size_t i = 0; i < args.size(); ++i) {
        if (func.getArgRole(i) == ArgRole::Param) varied_.insert(args[i].get());
    }
    const auto& ops = func.getEntryBlock()->getOperations();
    for (const auto& op : ops) {
        if (op->getNumResults() == 0) continue;
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            if (isDifferentiableOperand(*op, i) && isVaried(op->getOperand(i).get())) {
                varied_.insert(op->getResult(0).get());
                break;
            }
        }
    }
    if (loss) useful_.insert(loss);
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const Operation& op = **it;
        if (op.getNumResults() == 0 || !isUseful(op.getResult(0).get())) continue;
        for (size_t i = 0; i < op.getNumOperands(); ++i) {
            if (isDifferentiableOperand(op, i)) useful_.insert(op.getOperand(i).get());
        }
    }
}

bool ActivityPass::runOnFunction(Function& func) {
    const auto& ops = func.getEntryBlock()->getOperations();
    if (ops.empty() || ops.back()->getOpName() != "return" || ops.back()->getNumOperands() == 0) return false;
    ActivityAnalysis activity(func, ops.back()->getOperand(0).get());
    bool changed = false;
    for (const auto& op : ops) {
        if (op->getNumResults() > 0 && activity.isActive(op->getResult(0).get())) {
            op->setAttribute("ad.active", int64_t{1});
            changed = true;
        }
    }
    return changed;
}

} // namespace dlc
