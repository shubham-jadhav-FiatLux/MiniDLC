#include "dlc/ir/Operation.h"
#include <sstream>
#include <algorithm>

namespace dlc {

Operation::Operation(std::string opName,
                     std::vector<ValuePtr> operands,
                     std::vector<TypePtr> resultTypes)
    : opName_(std::move(opName)), operands_(std::move(operands)) {
    
    // Register uses with operands
    for (auto& op : operands_) {
        if (op) {
            op->addUse(this);
        }
    }

    // Allocate result SSA values
    static uint32_t sGlobalValId = 1000;
    for (size_t i = 0; i < resultTypes.size(); ++i) {
        auto val = std::make_shared<Value>(sGlobalValId++, resultTypes[i], this);
        results_.push_back(val);
    }
}

void Operation::setOperand(size_t idx, ValuePtr val) {
    if (idx < operands_.size()) {
        if (operands_[idx]) {
            operands_[idx]->removeUse(this);
        }
        operands_[idx] = val;
        if (val) {
            val->addUse(this);
        }
    }
}

std::string Operation::toString() const {
    std::ostringstream oss;
    if (!results_.empty()) {
        for (size_t i = 0; i < results_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << results_[i]->toString();
        }
        oss << " = ";
    }
    oss << opName_ << "(";
    for (size_t i = 0; i < operands_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << (operands_[i] ? operands_[i]->toString() : "%null");
    }
    oss << ")";
    if (!results_.empty()) {
        oss << " : ";
        for (size_t i = 0; i < results_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << results_[i]->getType()->toString();
        }
    }
    if (!attributes_.empty()) {
        oss << " {";
        bool first = true;
        // Use std::vector or sorted map for deterministic output
        std::vector<std::string> keys;
        for (const auto& [k, _] : attributes_) keys.push_back(k);
        std::sort(keys.begin(), keys.end());
        for (const auto& k : keys) {
            if (!first) oss << ", ";
            first = false;
            oss << k << " = ";
            std::visit([&oss](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    oss << "\"" << arg << "\"";
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    oss << arg;
                } else if constexpr (std::is_same_v<T, double>) {
                    oss << arg;
                } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                    oss << "[";
                    for (size_t i = 0; i < arg.size(); ++i) {
                        if (i > 0) oss << ", ";
                        oss << arg[i];
                    }
                    oss << "]";
                } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                    oss << "[";
                    for (size_t i = 0; i < arg.size(); ++i) {
                        if (i > 0) oss << ", ";
                        oss << arg[i];
                    }
                    oss << "]";
                }
            }, attributes_.at(k));
        }
        oss << "}";
    }
    return oss.str();
}

} // namespace dlc
