#pragma once

#include "dlc/ir/Operation.h"
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dlc {

enum class AttrKind { Int, Double, IntList, String };

// The kind of kernel an op lowers to.
enum class OpClass { Elementwise, Matmul, Reduction, RowReduction, Loss, Shape, Gather, Constant, Terminator };

struct AttrSpec {
    std::string name;
    AttrKind kind{AttrKind::Int};
    bool required{false};
};

// Everything the compiler knows about one op: shape inference, the verifier, the parser and the builder all read it.
struct OpDef {
    std::string name;
    int numOperands{0}; // -1: variadic
    bool hasResult{true};
    OpClass opClass{OpClass::Elementwise};
    std::vector<AttrSpec> attrs;
    // Adds optional attributes that are missing (e.g. softmax axis = -1, sum over all axes).
    std::function<void(Operation&)> setDefaults;
    // Computes the result type from the operands and attributes; throws IRError when they are invalid.
    std::function<TypePtr(const Operation&)> inferType;

    const AttrSpec* findAttr(std::string_view attrName) const;
};

class OpRegistry {
public:
    static const OpRegistry& instance();

    const OpDef* lookup(std::string_view name) const;
    const std::vector<OpDef>& ops() const { return ops_; }

private:
    OpRegistry();

    std::vector<OpDef> ops_;
    std::unordered_map<std::string, size_t> index_;
};

std::string opClassName(OpClass opClass);
std::string attrKindName(AttrKind kind);
bool attrHasKind(const AttributeValue& value, AttrKind kind);

// Adds default attributes and, when `inferResultType` is set, computes the op's result type from the registry.
void finalizeOp(Operation& op, bool inferResultType);

} // namespace dlc
