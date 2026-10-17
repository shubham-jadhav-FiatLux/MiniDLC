#pragma once

#include "dlc/ir/Operation.h"
#include <string>
#include <string_view>
#include <vector>
#include <memory>

namespace dlc {

class Function;

// What a function argument holds: per-step input data, a trainable parameter, a constant bound at compile time, or
// optimizer state that persists across steps.
enum class ArgRole { Input, Param, Const, State };

std::string argRoleName(ArgRole role);
bool parseArgRole(std::string_view text, ArgRole& role);

class Block {
public:
    explicit Block(Function* parentFunc = nullptr) : parentFunc_(parentFunc) {}

    Function* getParentFunction() const { return parentFunc_; }
    void setParentFunction(Function* func) { parentFunc_ = func; }

    const std::vector<OperationPtr>& getOperations() const { return operations_; }
    void addOperation(OperationPtr op);
    void insertOperation(size_t index, OperationPtr op);
    // Removes an op whose results are unused, unlinking it from its operands; throws IRError if a result is used.
    void eraseOperation(Operation* op);

private:
    Function* parentFunc_{nullptr};
    std::vector<OperationPtr> operations_;
};

class Function {
public:
    Function(std::string name, std::vector<ValuePtr> arguments, std::vector<TypePtr> resultTypes)
        : name_(std::move(name)), arguments_(std::move(arguments)), argRoles_(arguments_.size(), ArgRole::Input),
          resultTypes_(std::move(resultTypes)) {
        entryBlock_ = std::make_unique<Block>(this);
    }

    const std::string& getName() const { return name_; }
    const std::vector<ValuePtr>& getArguments() const { return arguments_; }
    ArgRole getArgRole(size_t idx) const { return argRoles_.at(idx); }
    void setArgRole(size_t idx, ArgRole role) { argRoles_.at(idx) = role; }
    const std::vector<TypePtr>& getResultTypes() const { return resultTypes_; }
    void setResultTypes(std::vector<TypePtr> resultTypes) { resultTypes_ = std::move(resultTypes); }

    Block* getEntryBlock() { return entryBlock_.get(); }
    const Block* getEntryBlock() const { return entryBlock_.get(); }

    std::string toString() const;

private:
    std::string name_;
    std::vector<ValuePtr> arguments_;
    std::vector<ArgRole> argRoles_;
    std::vector<TypePtr> resultTypes_;
    std::unique_ptr<Block> entryBlock_;
};

using FunctionPtr = std::shared_ptr<Function>;

} // namespace dlc
