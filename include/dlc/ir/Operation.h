#pragma once

#include "dlc/ir/Value.h"
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace dlc {

class Block;
class SlotTracker;

using AttributeValue = std::variant<int64_t, double, std::string, std::vector<int64_t>, std::vector<double>>;

class Operation {
public:
    Operation(std::string opName,
              std::vector<ValuePtr> operands = {},
              std::vector<TypePtr> resultTypes = {});
    virtual ~Operation();

    const std::string& getOpName() const { return opName_; }

    size_t getNumOperands() const { return operands_.size(); }
    ValuePtr getOperand(size_t idx) const { return operands_.at(idx); }
    const std::vector<ValuePtr>& getOperands() const { return operands_; }
    void setOperand(size_t idx, ValuePtr val);

    size_t getNumResults() const { return results_.size(); }
    ValuePtr getResult(size_t idx) const { return results_.at(idx); }
    const std::vector<ValuePtr>& getResults() const { return results_; }

    void setAttribute(const std::string& key, AttributeValue val) { attributes_[key] = std::move(val); }
    void removeAttribute(const std::string& key) { attributes_.erase(key); }
    bool hasAttribute(const std::string& key) const { return attributes_.find(key) != attributes_.end(); }
    const AttributeValue& getAttribute(const std::string& key) const { return attributes_.at(key); }
    const std::unordered_map<std::string, AttributeValue>& getAttributes() const { return attributes_; }

    // Typed attribute access; throws IRError when the attribute is missing or has another kind.
    int64_t getIntAttr(const std::string& key) const;
    double getDoubleAttr(const std::string& key) const;
    const std::vector<int64_t>& getIntListAttr(const std::string& key) const;
    const std::string& getStringAttr(const std::string& key) const;

    // Removes this op from its operands' use lists and clears its operands (done when the op is erased).
    void dropAllReferences();

    Block* getParentBlock() const { return parentBlock_; }
    void setParentBlock(Block* block) { parentBlock_ = block; }

    // Prints the op in textual IR form, naming values as `names` does.
    void print(std::ostream& os, const SlotTracker& names) const;
    // Prints the op with the values' own names; for diagnostics outside a function.
    virtual std::string toString() const;

protected:
    void printWith(std::ostream& os, const std::function<std::string(const Value*)>& nameOf) const;

    std::string opName_;
    std::vector<ValuePtr> operands_;
    std::vector<ValuePtr> results_;
    std::unordered_map<std::string, AttributeValue> attributes_;
    Block* parentBlock_{nullptr};
};

using OperationPtr = std::shared_ptr<Operation>;

// Points every use of `from` at `to`.
void replaceAllUsesWith(const ValuePtr& from, const ValuePtr& to);

// Prints an attribute value so that the parser reads back the same value.
void printAttributeValue(std::ostream& os, const AttributeValue& value);

} // namespace dlc
