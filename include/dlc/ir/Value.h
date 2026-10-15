#pragma once

#include "dlc/ir/Type.h"
#include <string>
#include <vector>
#include <memory>

namespace dlc {

class Operation;

class Value {
public:
    Value(uint32_t id, TypePtr type, Operation* definingOp = nullptr, std::string name = "")
        : id_(id), type_(std::move(type)), definingOp_(definingOp), name_(std::move(name)) {}

    uint32_t getId() const { return id_; }
    TypePtr getType() const { return type_; }
    void setType(TypePtr type) { type_ = std::move(type); }
    Operation* getDefiningOp() const { return definingOp_; }
    void setDefiningOp(Operation* op) { definingOp_ = op; }
    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    const std::vector<Operation*>& getUses() const { return uses_; }
    void addUse(Operation* op) { uses_.push_back(op); }
    void removeUse(Operation* op);

    std::string toString() const;

private:
    uint32_t id_;
    TypePtr type_;
    Operation* definingOp_;
    std::string name_;
    std::vector<Operation*> uses_;
};

using ValuePtr = std::shared_ptr<Value>;

} // namespace dlc
