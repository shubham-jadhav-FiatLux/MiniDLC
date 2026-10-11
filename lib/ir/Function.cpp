#include "dlc/ir/Function.h"
#include <algorithm>
#include <sstream>

namespace dlc {

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

void Block::removeOperation(Operation* op) {
    auto it = std::find_if(operations_.begin(), operations_.end(),
                           [op](const OperationPtr& ptr) { return ptr.get() == op; });
    if (it != operations_.end()) {
        operations_.erase(it);
    }
}

std::string Function::toString() const {
    std::ostringstream oss;
    oss << "func @" << name_ << "(";
    for (size_t i = 0; i < arguments_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << arguments_[i]->toString() << ": " << arguments_[i]->getType()->toString();
    }
    oss << ") -> ";
    if (resultTypes_.empty()) {
        oss << "()";
    } else if (resultTypes_.size() == 1) {
        oss << resultTypes_[0]->toString();
    } else {
        oss << "(";
        for (size_t i = 0; i < resultTypes_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << resultTypes_[i]->toString();
        }
        oss << ")";
    }
    oss << " {\n";
    if (entryBlock_) {
        for (const auto& op : entryBlock_->getOperations()) {
            oss << "  " << op->toString() << "\n";
        }
    }
    oss << "}\n";
    return oss.str();
}

} // namespace dlc
