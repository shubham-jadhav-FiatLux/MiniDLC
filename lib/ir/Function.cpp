#include "dlc/ir/Function.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/SlotTracker.h"
#include <algorithm>
#include <sstream>

namespace dlc {

std::string argRoleName(ArgRole role) {
    switch (role) {
        case ArgRole::Input: return "input";
        case ArgRole::Param: return "param";
        case ArgRole::Const: return "const";
        case ArgRole::State: return "state";
    }
    return "input";
}

std::string resultKindName(ResultKind kind) {
    switch (kind) {
        case ResultKind::Value: return "value";
        case ResultKind::Grad: return "grad";
        case ResultKind::Donates: return "donates";
    }
    return "value";
}

bool parseArgRole(std::string_view text, ArgRole& role) {
    for (ArgRole candidate : {ArgRole::Input, ArgRole::Param, ArgRole::Const, ArgRole::State}) {
        if (text == argRoleName(candidate)) {
            role = candidate;
            return true;
        }
    }
    return false;
}

void Block::addOperation(OperationPtr op) {
    if (op) {
        op->setParentBlock(this);
        operations_.push_back(std::move(op));
    }
}

void Block::insertOperation(size_t index, OperationPtr op) {
    if (op) {
        op->setParentBlock(this);
        if (index <= operations_.size()) {
            operations_.insert(operations_.begin() + index, std::move(op));
        } else {
            operations_.push_back(std::move(op));
        }
    }
}

void Block::eraseOperation(Operation* op) {
    auto it = std::find_if(operations_.begin(), operations_.end(),
                           [op](const OperationPtr& ptr) { return ptr.get() == op; });
    if (it == operations_.end()) return;
    for (const auto& result : op->getResults()) {
        if (!result->getUses().empty()) {
            throw IRError("cannot erase '" + op->getOpName() + "': its result is still used");
        }
    }
    op->dropAllReferences();
    op->setParentBlock(nullptr);
    operations_.erase(it);
}

ValuePtr Function::addArgument(TypePtr type, const std::string& name, ArgRole role, std::optional<double> init) {
    auto value = std::make_shared<Value>(Value::nextId(), std::move(type), nullptr, name);
    arguments_.push_back(value);
    argRoles_.push_back(role);
    argInits_.push_back(init);
    return value;
}

int64_t Function::getArgIndex(const Value* value) const {
    for (size_t i = 0; i < arguments_.size(); ++i) {
        if (arguments_[i].get() == value) return static_cast<int64_t>(i);
    }
    return -1;
}

void Function::setResults(std::vector<TypePtr> resultTypes, std::vector<ResultInfo> resultInfo) {
    if (resultTypes.size() != resultInfo.size()) throw IRError("setResults: one ResultInfo per result type");
    resultTypes_ = std::move(resultTypes);
    resultInfo_ = std::move(resultInfo);
}

std::string Function::toString() const {
    SlotTracker names(*this);
    std::ostringstream oss;
    oss << "func @" << name_ << "(";
    for (size_t i = 0; i < arguments_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "%" << names.name(arguments_[i].get()) << ": " << arguments_[i]->getType()->toString();
        if (argRoles_[i] != ArgRole::Input || argInits_[i]) {
            oss << " {" << argRoleName(argRoles_[i]);
            if (argInits_[i]) {
                oss << ", init = ";
                printAttributeValue(oss, *argInits_[i]);
            }
            oss << "}";
        }
    }
    oss << ") -> ";
    bool annotated = false;
    for (const auto& info : resultInfo_) annotated = annotated || info.kind != ResultKind::Value;
    if (resultTypes_.size() == 1 && !annotated) {
        oss << resultTypes_[0]->toString();
    } else {
        oss << "(";
        for (size_t i = 0; i < resultTypes_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << resultTypes_[i]->toString();
            const ResultInfo& info = resultInfo_[i];
            if (info.kind != ResultKind::Value) {
                oss << " {" << resultKindName(info.kind) << " = %" << names.name(arguments_.at(info.arg).get()) << "}";
            }
        }
        oss << ")";
    }
    oss << " {\n";
    for (const auto& op : entryBlock_->getOperations()) {
        oss << "  ";
        op->print(oss, names);
        oss << "\n";
    }
    oss << "}\n";
    return oss.str();
}

} // namespace dlc
