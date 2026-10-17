#include "dlc/passes/Canonicalize.h"
#include "dlc/ir/Ops.h"

namespace dlc {

static bool isConstantValue(const ValuePtr& val, double& outVal) {
    const Operation* def = val ? val->getDefiningOp() : nullptr;
    if (!def || def->getOpName() != "constant") return false;
    outVal = def->getDoubleAttr("value");
    return true;
}

bool CanonicalizePass::runOnFunction(Function& func) {
    bool changed = false;
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    // Pattern simplifications & constant folding. Iterate over a snapshot: folding inserts new ops into the block.
    const std::vector<OperationPtr> ops = entry->getOperations();
    for (const auto& op : ops) {
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
                replaceAllUsesWith(res, folded->getResult(0));
                changed = true;
            } else if (isCR && vR == 0.0) {
                replaceAllUsesWith(res, lhs);
                changed = true;
            } else if (isCL && vL == 0.0) {
                replaceAllUsesWith(res, rhs);
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
                replaceAllUsesWith(res, folded->getResult(0));
                changed = true;
            } else if (isCR && vR == 0.0) {
                replaceAllUsesWith(res, lhs);
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
                replaceAllUsesWith(res, folded->getResult(0));
                changed = true;
            } else if ((isCR && vR == 1.0)) {
                replaceAllUsesWith(res, lhs);
                changed = true;
            } else if ((isCL && vL == 1.0)) {
                replaceAllUsesWith(res, rhs);
                changed = true;
            } else if ((isCR && vR == 0.0)) {
                replaceAllUsesWith(res, rhs);
                changed = true;
            } else if ((isCL && vL == 0.0)) {
                replaceAllUsesWith(res, lhs);
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
                replaceAllUsesWith(res, folded->getResult(0));
                changed = true;
            } else if (isCR && vR == 1.0) {
                replaceAllUsesWith(res, lhs);
                changed = true;
            }
        }
    }

    return changed;
}

} // namespace dlc
