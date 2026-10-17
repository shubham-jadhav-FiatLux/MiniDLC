#include "dlc/ir/Verifier.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/OpRegistry.h"
#include "dlc/ir/SlotTracker.h"
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace dlc {

namespace {

std::string typeString(const TypePtr& type) { return type ? type->toString() : "<no type>"; }

// Registry, arity, attribute and type checks for one op; `describe` labels the op in messages.
void verifyOp(const Operation& op, const std::string& label, VerificationResult& res) {
    const OpDef* def = OpRegistry::instance().lookup(op.getOpName());
    if (!def) {
        res.addError(label + ": unknown op '" + op.getOpName() + "'");
        return;
    }
    if (def->numOperands >= 0 && op.getNumOperands() != static_cast<size_t>(def->numOperands)) {
        res.addError(label + ": expects " + std::to_string(def->numOperands) + " operand(s), has " +
                     std::to_string(op.getNumOperands()));
        return;
    }
    for (size_t i = 0; i < op.getNumOperands(); ++i) {
        if (!op.getOperand(i)) {
            res.addError(label + ": operand " + std::to_string(i) + " is null");
            return;
        }
    }
    size_t expectedResults = def->hasResult ? 1 : 0;
    if (op.getNumResults() != expectedResults) {
        res.addError(label + ": expects " + std::to_string(expectedResults) + " result(s), has " +
                     std::to_string(op.getNumResults()));
        return;
    }

    bool attrsOk = true;
    for (const auto& spec : def->attrs) {
        if (!op.hasAttribute(spec.name)) {
            if (spec.required) {
                res.addError(label + ": missing required attribute '" + spec.name + "'");
                attrsOk = false;
            }
        } else if (!attrHasKind(op.getAttribute(spec.name), spec.kind)) {
            res.addError(label + ": attribute '" + spec.name + "' must be " + attrKindName(spec.kind));
            attrsOk = false;
        }
    }
    for (const auto& [key, _] : op.getAttributes()) {
        if (!def->findAttr(key) && key.find('.') == std::string::npos) {
            res.addError(label + ": unknown attribute '" + key + "'");
            attrsOk = false;
        }
    }

    if (def->hasResult && attrsOk) {
        try {
            TypePtr inferred = def->inferType(op);
            const TypePtr& actual = op.getResult(0)->getType();
            if (typeString(inferred) != typeString(actual)) {
                res.addError(label + ": result type " + typeString(actual) + " does not match the inferred type " +
                             typeString(inferred));
            }
        } catch (const IRError& e) {
            res.addError(label + ": " + e.what());
        }
    }
}

} // namespace

VerificationResult Verifier::verify(const Module& module) {
    VerificationResult res;
    for (const auto& func : module.getFunctions()) {
        auto funcRes = verify(*func);
        for (auto& err : funcRes.errors) {
            res.addError("[@" + func->getName() + "] " + err);
        }
    }
    return res;
}

VerificationResult Verifier::verify(const Function& func) {
    VerificationResult res;
    const Block* entry = func.getEntryBlock();
    if (!entry) {
        res.addError("function has no entry block");
        return res;
    }
    SlotTracker names(func);
    const auto& ops = entry->getOperations();

    std::unordered_set<const Value*> defined;
    for (const auto& arg : func.getArguments()) defined.insert(arg.get());

    size_t returnCount = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
        const Operation& op = *ops[i];
        std::ostringstream text;
        op.print(text, names);
        std::string label = "'" + text.str() + "'";

        for (size_t j = 0; j < op.getNumOperands(); ++j) {
            const Value* operand = op.getOperand(j).get();
            if (operand && !defined.count(operand)) {
                res.addError(label + ": operand " + std::to_string(j) + " is used before it is defined");
            }
        }
        verifyOp(op, label, res);
        for (const auto& result : op.getResults()) defined.insert(result.get());

        if (op.getOpName() == "return") {
            ++returnCount;
            if (i + 1 != ops.size()) res.addError(label + ": 'return' must be the last op");
            const auto& resultTypes = func.getResultTypes();
            if (op.getNumOperands() != resultTypes.size()) {
                res.addError(label + ": returns " + std::to_string(op.getNumOperands()) +
                             " value(s) but the function declares " + std::to_string(resultTypes.size()));
            } else {
                for (size_t j = 0; j < resultTypes.size(); ++j) {
                    const ValuePtr& value = op.getOperand(j);
                    if (value && typeString(value->getType()) != typeString(resultTypes[j])) {
                        res.addError(label + ": returned value " + std::to_string(j) + " has type " +
                                     typeString(value->getType()) + " but the function declares " +
                                     typeString(resultTypes[j]));
                    }
                }
            }
        }
    }
    if (returnCount == 0) res.addError("function has no 'return'");
    if (returnCount > 1) res.addError("function has " + std::to_string(returnCount) + " 'return' ops");

    const auto& args = func.getArguments();
    for (size_t i = 0; i < args.size(); ++i) {
        if (func.getArgInit(i) && func.getArgRole(i) != ArgRole::State) {
            res.addError("argument '%" + names.name(args[i].get()) + "': only state arguments have an initial value");
        }
    }
    std::vector<bool> donated(args.size(), false);
    const auto& resultTypes = func.getResultTypes();
    const auto& resultInfo = func.getResultInfo();
    for (size_t i = 0; i < resultInfo.size() && i < resultTypes.size(); ++i) {
        const ResultInfo& info = resultInfo[i];
        if (info.kind == ResultKind::Value) continue;
        std::string label = "result " + std::to_string(i) + " (" + resultKindName(info.kind) + ")";
        if (info.arg >= args.size()) {
            res.addError(label + " names argument " + std::to_string(info.arg) + ", which does not exist");
            continue;
        }
        const std::string argName = "'%" + names.name(args[info.arg].get()) + "'";
        if (typeString(resultTypes[i]) != typeString(args[info.arg]->getType())) {
            res.addError(label + " has type " + typeString(resultTypes[i]) + " but " + argName + " is " +
                         typeString(args[info.arg]->getType()));
        }
        ArgRole role = func.getArgRole(info.arg);
        if (info.kind == ResultKind::Grad && role != ArgRole::Param) {
            res.addError(label + ": " + argName + " is not a param argument");
        }
        if (info.kind == ResultKind::Donates) {
            if (role != ArgRole::Param && role != ArgRole::State) {
                res.addError(label + ": only param and state arguments can be donated, not " + argName);
            }
            if (donated[info.arg]) res.addError(label + ": " + argName + " is donated twice");
            donated[info.arg] = true;
        }
    }

    // Every value's use list must name exactly the ops of this function that use it.
    std::unordered_map<const Value*, std::unordered_map<const Operation*, int>> expected;
    for (const auto& op : ops) {
        for (const auto& operand : op->getOperands()) {
            if (operand) ++expected[operand.get()][op.get()];
        }
    }
    auto checkUses = [&](const Value* value) {
        std::unordered_map<const Operation*, int> actual;
        for (const Operation* user : value->getUses()) ++actual[user];
        if (actual != expected[value]) {
            res.addError("use list of '%" + names.name(value) + "' does not match the ops that use it");
        }
    };
    for (const auto& arg : func.getArguments()) checkUses(arg.get());
    for (const auto& op : ops) {
        for (const auto& result : op->getResults()) checkUses(result.get());
    }
    return res;
}

VerificationResult Verifier::verify(const Operation& op) {
    VerificationResult res;
    verifyOp(op, "'" + op.toString() + "'", res);
    return res;
}

} // namespace dlc
