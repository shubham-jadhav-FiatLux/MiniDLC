#include "dlc/ir/Operation.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/SlotTracker.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <sstream>

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
    for (size_t i = 0; i < resultTypes.size(); ++i) {
        results_.push_back(std::make_shared<Value>(Value::nextId(), resultTypes[i], this));
    }
}

Operation::~Operation() {
    dropAllReferences();
    for (auto& result : results_) {
        if (result && result->getDefiningOp() == this) result->setDefiningOp(nullptr);
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

void Operation::dropAllReferences() {
    for (auto& operand : operands_) {
        if (operand) operand->removeUse(this);
    }
    operands_.clear();
}

namespace {

template <typename T>
const T& attributeAs(const Operation& op, const std::string& key, const char* kind) {
    auto it = op.getAttributes().find(key);
    if (it == op.getAttributes().end()) {
        throw IRError("'" + op.getOpName() + "' is missing attribute '" + key + "'");
    }
    if (const T* value = std::get_if<T>(&it->second)) return *value;
    throw IRError("attribute '" + key + "' of '" + op.getOpName() + "' must be " + kind);
}

// The shortest text that reads back as the same double, laid out like Python's repr(): scientific notation when the
// decimal exponent is below -4 or above 15, otherwise positional with at least one fractional digit. The Python
// exporter prints the same way, so both sides produce identical IR text.
void printDouble(std::ostream& os, double value) {
    if (std::isnan(value)) {
        os << "nan";
        return;
    }
    if (std::isinf(value)) {
        os << (value < 0 ? "-inf" : "inf");
        return;
    }
    char buffer[64];
    auto [end, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
    std::string text(buffer, end); // e.g. "-1.25e+02"
    size_t ePos = text.find('e');
    int exponent = std::stoi(text.substr(ePos + 1));
    if (exponent < -4 || exponent >= 16) {
        os << text;
        return;
    }
    bool negative = text[0] == '-';
    std::string digits;
    for (size_t i = negative ? 1 : 0; i < ePos; ++i) {
        if (text[i] != '.') digits += text[i];
    }
    int point = exponent + 1; // digits before the decimal point
    std::string fixed;
    if (point <= 0) {
        fixed = "0." + std::string(-point, '0') + digits;
    } else if (point < static_cast<int>(digits.size())) {
        fixed = digits.substr(0, point) + "." + digits.substr(point);
    } else {
        fixed = digits + std::string(point - digits.size(), '0') + ".0";
    }
    os << (negative ? "-" : "") << fixed;
}

void printQuoted(std::ostream& os, const std::string& text) {
    os << '"';
    for (char c : text) {
        if (c == '"' || c == '\\') os << '\\' << c;
        else if (c == '\n') os << "\\n";
        else if (c == '\t') os << "\\t";
        else os << c;
    }
    os << '"';
}

} // namespace

int64_t Operation::getIntAttr(const std::string& key) const {
    return attributeAs<int64_t>(*this, key, "an integer");
}

double Operation::getDoubleAttr(const std::string& key) const {
    return attributeAs<double>(*this, key, "a float");
}

const std::vector<int64_t>& Operation::getIntListAttr(const std::string& key) const {
    return attributeAs<std::vector<int64_t>>(*this, key, "a list of integers");
}

const std::string& Operation::getStringAttr(const std::string& key) const {
    return attributeAs<std::string>(*this, key, "a string");
}

void printAttributeValue(std::ostream& os, const AttributeValue& value) {
    std::visit([&os](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) {
            os << arg;
        } else if constexpr (std::is_same_v<T, double>) {
            printDouble(os, arg);
        } else if constexpr (std::is_same_v<T, std::string>) {
            printQuoted(os, arg);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            os << "[";
            for (size_t i = 0; i < arg.size(); ++i) os << (i ? ", " : "") << arg[i];
            os << "]";
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            os << "[";
            for (size_t i = 0; i < arg.size(); ++i) {
                if (i) os << ", ";
                printDouble(os, arg[i]);
            }
            os << "]";
        }
    }, value);
}

void Operation::printWith(std::ostream& os, const std::function<std::string(const Value*)>& nameOf) const {
    if (opName_ == "return") {
        os << "return";
        for (size_t i = 0; i < operands_.size(); ++i) os << (i ? ", %" : " %") << nameOf(operands_[i].get());
        return;
    }
    for (size_t i = 0; i < results_.size(); ++i) os << (i ? ", %" : "%") << nameOf(results_[i].get());
    if (!results_.empty()) os << " = ";
    os << opName_ << "(";
    for (size_t i = 0; i < operands_.size(); ++i) os << (i ? ", %" : "%") << nameOf(operands_[i].get());
    os << ")";
    for (size_t i = 0; i < results_.size(); ++i) {
        const TypePtr& type = results_[i]->getType();
        os << (i ? ", " : " : ") << (type ? type->toString() : "<no type>");
    }
    if (!attributes_.empty()) {
        std::vector<std::string> keys;
        for (const auto& [key, _] : attributes_) keys.push_back(key);
        std::sort(keys.begin(), keys.end());
        os << " {";
        for (size_t i = 0; i < keys.size(); ++i) {
            os << (i ? ", " : "") << keys[i] << " = ";
            printAttributeValue(os, attributes_.at(keys[i]));
        }
        os << "}";
    }
}

void Operation::print(std::ostream& os, const SlotTracker& names) const {
    printWith(os, [&names](const Value* value) { return names.name(value); });
}

std::string Operation::toString() const {
    std::ostringstream oss;
    printWith(oss, [](const Value* value) {
        if (!value) return std::string("<null>");
        std::string text = value->toString();
        return text.substr(1); // drop the '%' that Value::toString adds
    });
    return oss.str();
}

void replaceAllUsesWith(const ValuePtr& from, const ValuePtr& to) {
    if (!from || !to || from == to) return;
    std::vector<Operation*> users = from->getUses(); // copy: setOperand edits the list
    for (Operation* user : users) {
        for (size_t i = 0; i < user->getNumOperands(); ++i) {
            if (user->getOperand(i) == from) user->setOperand(i, to);
        }
    }
}

} // namespace dlc
