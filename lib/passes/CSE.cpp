#include "dlc/passes/CSE.h"
#include <algorithm>
#include <bit>
#include <sstream>
#include <unordered_map>

namespace dlc {

namespace {

void writeExact(std::ostream& os, const AttributeValue& value) {
    std::visit([&os](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) {
            os << 'i' << arg;
        } else if constexpr (std::is_same_v<T, double>) {
            os << 'd' << std::bit_cast<uint64_t>(arg);
        } else if constexpr (std::is_same_v<T, std::string>) {
            os << 's' << arg.size() << ':' << arg;
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            os << 'I' << arg.size();
            for (int64_t v : arg) os << ',' << v;
        } else {
            os << 'D' << arg.size();
            for (double v : arg) os << ',' << std::bit_cast<uint64_t>(v);
        }
    }, value);
}

std::string valueKey(const Operation& op) {
    std::ostringstream oss;
    oss << op.getOpName() << '(';
    for (const auto& operand : op.getOperands()) oss << static_cast<const void*>(operand.get()) << ',';
    oss << ')';
    for (const auto& result : op.getResults()) oss << ':' << (result->getType() ? result->getType()->toString() : "?");
    std::vector<std::string> keys;
    for (const auto& [key, _] : op.getAttributes()) keys.push_back(key);
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) {
        oss << '{' << key << '=';
        writeExact(oss, op.getAttribute(key));
        oss << '}';
    }
    return oss.str();
}

} // namespace

bool CSEPass::runOnFunction(Function& func) {
    Block* entry = func.getEntryBlock();
    std::unordered_map<std::string, ValuePtr> available;
    std::vector<Operation*> duplicates;
    for (const auto& op : entry->getOperations()) {
        if (op->getNumResults() != 1) continue;
        auto [it, inserted] = available.emplace(valueKey(*op), op->getResult(0));
        if (inserted) continue;
        replaceAllUsesWith(op->getResult(0), it->second);
        duplicates.push_back(op.get());
    }
    for (Operation* op : duplicates) entry->eraseOperation(op);
    return !duplicates.empty();
}

} // namespace dlc
