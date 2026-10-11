#include "dlc/ir/Module.h"
#include <sstream>

namespace dlc {

void Module::addFunction(FunctionPtr func) {
    if (func) {
        functionMap_[func->getName()] = func;
        functions_.push_back(std::move(func));
    }
}

FunctionPtr Module::getFunction(const std::string& name) const {
    auto it = functionMap_.find(name);
    if (it != functionMap_.end()) {
        return it->second;
    }
    return nullptr;
}

std::string Module::toString() const {
    std::ostringstream oss;
    oss << "// Module: " << name_ << "\n\n";
    for (const auto& func : functions_) {
        oss << func->toString() << "\n";
    }
    return oss.str();
}

} // namespace dlc
