#pragma once

#include "dlc/ir/Value.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <variant>

namespace dlc {

class Block;

using AttributeValue = std::variant<int64_t, double, std::string, std::vector<int64_t>, std::vector<double>>;

class Operation {
public:
    Operation(std::string opName,
              std::vector<ValuePtr> operands = {},
              std::vector<TypePtr> resultTypes = {});
    virtual ~Operation() = default;

    const std::string& getOpName() const { return opName_; }
    
    size_t getNumOperands() const { return operands_.size(); }
    ValuePtr getOperand(size_t idx) const { return operands_.at(idx); }
    const std::vector<ValuePtr>& getOperands() const { return operands_; }
    void setOperand(size_t idx, ValuePtr val);

    size_t getNumResults() const { return results_.size(); }
    ValuePtr getResult(size_t idx) const { return results_.at(idx); }
    const std::vector<ValuePtr>& getResults() const { return results_; }

    void setAttribute(const std::string& key, AttributeValue val) { attributes_[key] = std::move(val); }
    bool hasAttribute(const std::string& key) const { return attributes_.find(key) != attributes_.end(); }
    const AttributeValue& getAttribute(const std::string& key) const { return attributes_.at(key); }
    const std::unordered_map<std::string, AttributeValue>& getAttributes() const { return attributes_; }

    Block* getParentBlock() const { return parentBlock_; }
    void setParentBlock(Block* block) { parentBlock_ = block; }

    virtual std::string toString() const;

protected:
    std::string opName_;
    std::vector<ValuePtr> operands_;
    std::vector<ValuePtr> results_;
    std::unordered_map<std::string, AttributeValue> attributes_;
    Block* parentBlock_{nullptr};
};

using OperationPtr = std::shared_ptr<Operation>;

} // namespace dlc
