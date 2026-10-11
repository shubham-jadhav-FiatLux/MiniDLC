#include "dlc/ir/Verifier.h"
#include <unordered_set>

namespace dlc {

VerificationResult Verifier::verify(const Module& module) {
    VerificationResult res;
    for (const auto& func : module.getFunctions()) {
        auto funcRes = verify(*func);
        if (!funcRes.success) {
            res.success = false;
            for (auto& err : funcRes.errors) {
                res.addError("[" + func->getName() + "] " + err);
            }
        }
    }
    return res;
}

VerificationResult Verifier::verify(const Function& func) {
    VerificationResult res;
    std::unordered_set<Value*> definedValues;

    for (const auto& arg : func.getArguments()) {
        definedValues.insert(arg.get());
    }

    const Block* entry = func.getEntryBlock();
    if (!entry) {
        res.addError("Function missing entry block");
        return res;
    }

    bool hasTerminator = false;
    for (const auto& op : entry->getOperations()) {
        // Check dominance: all operands must have been defined prior
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            Value* opVal = op->getOperand(i).get();
            if (!opVal) {
                res.addError("Operation " + op->getOpName() + " has null operand at index " + std::to_string(i));
            } else if (definedValues.find(opVal) == definedValues.end()) {
                res.addError("Operand " + opVal->toString() + " used before definition in " + op->getOpName());
            }
        }

        // Add results to defined set
        for (size_t i = 0; i < op->getNumResults(); ++i) {
            definedValues.insert(op->getResult(i).get());
        }

        if (op->getOpName() == "return") {
            hasTerminator = true;
        }
    }

    if (!hasTerminator && !entry->getOperations().empty()) {
        res.addError("Function block missing return terminator");
    }

    return res;
}

VerificationResult Verifier::verify(const Operation& op) {
    VerificationResult res;
    if (op.getOpName().empty()) {
        res.addError("Operation has empty opName");
    }
    return res;
}

} // namespace dlc
