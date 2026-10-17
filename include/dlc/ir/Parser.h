#pragma once

#include "dlc/ir/Module.h"
#include <string_view>

namespace dlc {

// Parses textual graph IR (docs/ir-spec.md). Malformed input raises IRError with a "line:col: message" description.
class IRParser {
public:
    static ModulePtr parseModule(std::string_view source);
    // The first function of the module.
    static FunctionPtr parseFunction(std::string_view source);
};

} // namespace dlc
