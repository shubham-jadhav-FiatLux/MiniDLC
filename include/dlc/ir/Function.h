#pragma once

#include "dlc/ir/Operation.h"
#include <string>
#include <vector>
#include <memory>

namespace dlc {

class Function;

class Block {
public:
    explicit Block(Function* parentFunc = nullptr) : parentFunc_(parentFunc) {}

    Function* getParentFunction() const { return parentFunc_; }
    void setParentFunction(Function* func) { parentFunc_ = func; }

    const std::vector<OperationPtr>& getOperations() const { return operations_; }
    void addOperation(OperationPtr op);
    void insertOperation(size_t index, OperationPtr op);
    void removeOperation(Operation* op);

private:
    Function* parentFunc_{nullptr};
    std::vector<OperationPtr> operations_;
};

class Function {
public:
    Function(std::string name, std::vector<ValuePtr> arguments, std::vector<TypePtr> resultTypes)
        : name_(std::move(name)), arguments_(std::move(arguments)), resultTypes_(std::move(resultTypes)) {
        entryBlock_ = std::make_unique<Block>(this);
    }

    const std::string& getName() const { return name_; }
    const std::vector<ValuePtr>& getArguments() const { return arguments_; }
    const std::vector<TypePtr>& getResultTypes() const { return resultTypes_; }

    Block* getEntryBlock() { return entryBlock_.get(); }
    const Block* getEntryBlock() const { return entryBlock_.get(); }

    std::string toString() const;

private:
    std::string name_;
    std::vector<ValuePtr> arguments_;
    std::vector<TypePtr> resultTypes_;
    std::unique_ptr<Block> entryBlock_;
};

using FunctionPtr = std::shared_ptr<Function>;

} // namespace dlc
