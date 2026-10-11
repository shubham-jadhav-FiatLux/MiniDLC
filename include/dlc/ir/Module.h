#pragma once

#include "dlc/ir/Function.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace dlc {

class Module {
public:
    explicit Module(std::string name = "main") : name_(std::move(name)) {}

    const std::string& getName() const { return name_; }

    void addFunction(FunctionPtr func);
    FunctionPtr getFunction(const std::string& name) const;
    const std::vector<FunctionPtr>& getFunctions() const { return functions_; }

    std::string toString() const;

private:
    std::string name_;
    std::vector<FunctionPtr> functions_;
    std::unordered_map<std::string, FunctionPtr> functionMap_;
};

using ModulePtr = std::shared_ptr<Module>;

} // namespace dlc
