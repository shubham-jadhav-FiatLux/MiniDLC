#include "dlc/passes/Canonicalize.h"
#include "dlc/ir/Ops.h"

namespace dlc {

static void replaceAllUses(ValuePtr from, ValuePtr to) {
    if (!from || !to || from == to) return;
    auto uses = from->getUses();
    for (auto* user : uses) {
        for (size_t i = 0; i < user->getNumOperands(); ++i) {
            if (user->getOperand(i) == from) {
                user->setOperand(i, to);
            }
        }
    }
}

static bool isConstantValue(ValuePtr val, double& outVal) {
    if (val && val->getDefiningOp() && val->getDefiningOp()->getOpName() == "constant") {
        auto* cop = dynamic_cast<ConstantOp*>(val->getDefiningOp());
        if (cop) {
            outVal = cop->getValue();
            return true;
        }
    }
    return false;
}

bool CanonicalizePass::runOnFunction(Function& func) {
    bool changed = false;
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    // Pattern simplifications & constant folding
    for (auto& op : entry->getOperations()) {
        const std::string& name = op->getOpName();
        if (op->getNumResults() == 0) continue;
        ValuePtr res = op->getResult(0);

        if (name == "add" && op->getNumOperands() == 2) {
            ValuePtr lhs = op->getOperand(0);
            ValuePtr rhs = op->getOperand(1);
            double vL = 0.0, vR = 0.0;
            bool isCL = isConstantValue(lhs, vL);
            bool isCR = isConstantValue(rhs, vR);

            if (isCL && isCR) {
                // Constant fold
                auto folded = std::make_shared<ConstantOp>(vL + vR, res->getType());
                entry->insertOperation(0, folded);
                replaceAllUses(res, folded->getResult(0));
                changed = true;
            } else if (isCR && vR == 0.0) {
                replaceAllUses(res, lhs);
                changed = true;
            } else if (isCL && vL == 0.0) {
                replaceAllUses(res, rhs);
                changed = true;
            }
        } else if (name == "sub" && op->getNumOperands() == 2) {
            ValuePtr lhs = op->getOperand(0);
            ValuePtr rhs = op->getOperand(1);
            double vL = 0.0, vR = 0.0;
            bool isCL = isConstantValue(lhs, vL);
            bool isCR = isConstantValue(rhs, vR);

            if (isCL && isCR) {
                auto folded = std::make_shared<ConstantOp>(vL - vR, res->getType());
                entry->insertOperation(0, folded);
                replaceAllUses(res, folded->getResult(0));
                changed = true;
            } else if (isCR && vR == 0.0) {
                replaceAllUses(res, lhs);
                changed = true;
            }
        } else if (name == "mul" && op->getNumOperands() == 2) {
            ValuePtr lhs = op->getOperand(0);
            ValuePtr rhs = op->getOperand(1);
            double vL = 0.0, vR = 0.0;
            bool isCL = isConstantValue(lhs, vL);
            bool isCR = isConstantValue(rhs, vR);

            if (isCL && isCR) {
                auto folded = std::make_shared<ConstantOp>(vL * vR, res->getType());
                entry->insertOperation(0, folded);
                replaceAllUses(res, folded->getResult(0));
                changed = true;
            } else if ((isCR && vR == 1.0)) {
                replaceAllUses(res, lhs);
                changed = true;
            } else if ((isCL && vL == 1.0)) {
                replaceAllUses(res, rhs);
                changed = true;
            } else if ((isCR && vR == 0.0)) {
                replaceAllUses(res, rhs);
                changed = true;
            } else if ((isCL && vL == 0.0)) {
                replaceAllUses(res, lhs);
                changed = true;
            }
        } else if (name == "div" && op->getNumOperands() == 2) {
            ValuePtr lhs = op->getOperand(0);
            ValuePtr rhs = op->getOperand(1);
            double vL = 0.0, vR = 0.0;
            bool isCL = isConstantValue(lhs, vL);
            bool isCR = isConstantValue(rhs, vR);

            if (isCL && isCR && vR != 0.0) {
                auto folded = std::make_shared<ConstantOp>(vL / vR, res->getType());
                entry->insertOperation(0, folded);
                replaceAllUses(res, folded->getResult(0));
                changed = true;
            } else if (isCR && vR == 1.0) {
                replaceAllUses(res, lhs);
                changed = true;
            }
        }
    }

    return changed;
}

} // namespace dlc
