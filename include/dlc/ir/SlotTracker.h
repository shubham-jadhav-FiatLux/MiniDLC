#pragma once

#include "dlc/ir/Function.h"
#include <string>
#include <unordered_map>

namespace dlc {

// The names a function's values print with (without the '%' sigil). Arguments keep their own names when those are
// valid and unique, otherwise argN; op results are numbered 0, 1, ... in program order, so printing is deterministic.
class SlotTracker {
public:
    explicit SlotTracker(const Function& func);

    std::string name(const Value* value) const;

private:
    std::unordered_map<const Value*, std::string> names_;
};

// A name usable for an argument: a letter or '_' followed by letters, digits, '_', '.' or '$'.
bool isValidArgumentName(const std::string& name);

} // namespace dlc
