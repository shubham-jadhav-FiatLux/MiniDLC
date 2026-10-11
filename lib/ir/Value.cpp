#include "dlc/ir/Value.h"
#include <algorithm>
#include <sstream>

namespace dlc {

void Value::removeUse(Operation* op) {
    auto it = std::find(uses_.begin(), uses_.end(), op);
    if (it != uses_.end()) {
        uses_.erase(it);
    }
}

std::string Value::toString() const {
    std::ostringstream oss;
    if (!name_.empty()) {
        oss << "%" << name_;
    } else {
        oss << "%" << id_;
    }
    return oss.str();
}

} // namespace dlc
