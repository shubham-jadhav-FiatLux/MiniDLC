#pragma once

#include "dlc/ir/Operation.h"
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dlc {

class Function;

// What a function argument holds: per-step input data, a trainable parameter, a constant bound at compile time, or
// optimizer state that persists across steps.
enum class ArgRole { Input, Param, Const, State };

std::string argRoleName(ArgRole role);
bool parseArgRole(std::string_view text, ArgRole& role);

// What a function result is: a plain value, the gradient of an argument (autodiff), or the new value of a param or
// state argument that takes over the argument's buffer (optimizer lowering).
enum class ResultKind { Value, Grad, Donates };

struct ResultInfo {
    ResultKind kind{ResultKind::Value};
    size_t arg{0}; // the argument, for Grad and Donates
};

std::string resultKindName(ResultKind kind);

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
          argInits_(arguments_.size()), resultTypes_(std::move(resultTypes)), resultInfo_(resultTypes_.size()) {
        entryBlock_ = std::make_unique<Block>(this);
    }

    const std::string& getName() const { return name_; }
    const std::vector<ValuePtr>& getArguments() const { return arguments_; }
    ArgRole getArgRole(size_t idx) const { return argRoles_.at(idx); }
    void setArgRole(size_t idx, ArgRole role) { argRoles_.at(idx) = role; }
    // The value a state argument starts from (every element), e.g. 0 for Adam moments.
    const std::optional<double>& getArgInit(size_t idx) const { return argInits_.at(idx); }
    void setArgInit(size_t idx, std::optional<double> init) { argInits_.at(idx) = init; }
    ValuePtr addArgument(TypePtr type, const std::string& name, ArgRole role, std::optional<double> init = {});
    // The index of `value` among the arguments, or -1.
    int64_t getArgIndex(const Value* value) const;

    const std::vector<TypePtr>& getResultTypes() const { return resultTypes_; }
    // Replaces the results' types; every result becomes a plain value.
    void setResultTypes(std::vector<TypePtr> resultTypes) {
        resultTypes_ = std::move(resultTypes);
        resultInfo_.assign(resultTypes_.size(), ResultInfo{});
    }
    const std::vector<ResultInfo>& getResultInfo() const { return resultInfo_; }
    void setResults(std::vector<TypePtr> resultTypes, std::vector<ResultInfo> resultInfo);

    Block* getEntryBlock() { return entryBlock_.get(); }
    const Block* getEntryBlock() const { return entryBlock_.get(); }

    std::string toString() const;

private:
    std::string name_;
    std::vector<ValuePtr> arguments_;
    std::vector<ArgRole> argRoles_;
    std::vector<std::optional<double>> argInits_;
    std::vector<TypePtr> resultTypes_;
    std::vector<ResultInfo> resultInfo_;
    std::unique_ptr<Block> entryBlock_;
};

using FunctionPtr = std::shared_ptr<Function>;

} // namespace dlc
