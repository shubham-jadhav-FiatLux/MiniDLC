#include "dlc/ir/SlotTracker.h"
#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace dlc {

bool isValidArgumentName(const std::string& name) {
    if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) || name[0] == '_')) return false;
    return std::all_of(name.begin(), name.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '$';
    });
}

SlotTracker::SlotTracker(const Function& func) {
    const auto& args = func.getArguments();
    std::unordered_set<std::string> taken;
    std::vector<bool> named(args.size(), false);
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& own = args[i]->getName();
        if (isValidArgumentName(own) && taken.insert(own).second) {
            names_[args[i].get()] = own;
            named[i] = true;
        }
    }
    for (size_t i = 0; i < args.size(); ++i) {
        if (named[i]) continue;
        std::string fallback = "arg" + std::to_string(i);
        for (int k = 1; taken.count(fallback); ++k) fallback = "arg" + std::to_string(i) + "_" + std::to_string(k);
        taken.insert(fallback);
        names_[args[i].get()] = fallback;
    }

    int64_t next = 0;
    if (const Block* entry = func.getEntryBlock()) {
        for (const auto& op : entry->getOperations()) {
            for (const auto& result : op->getResults()) names_[result.get()] = std::to_string(next++);
        }
    }
}

std::string SlotTracker::name(const Value* value) const {
    if (!value) return "<null>";
    auto it = names_.find(value);
    if (it != names_.end()) return it->second;
    return "<undefined:" + std::to_string(value->getId()) + ">";
}

} // namespace dlc
