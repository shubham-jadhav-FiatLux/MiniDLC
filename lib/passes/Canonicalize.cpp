#include "dlc/passes/Canonicalize.h"
#include "dlc/ir/Builder.h"
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>

namespace dlc {

namespace {

// The value of a `constant` op (constants are splats), if `value` is one.
std::optional<double> splat(const ValuePtr& value) {
    const Operation* def = value ? value->getDefiningOp() : nullptr;
    if (!def || def->getOpName() != "constant") return std::nullopt;
    return def->getDoubleAttr("value");
}

const TensorType& tensor(const ValuePtr& value) { return static_cast<const TensorType&>(*value->getType()); }

bool sameType(const ValuePtr& a, const ValuePtr& b) { return a->getType()->toString() == b->getType()->toString(); }

double relu(double x) { return std::isnan(x) || x > 0 ? x : 0.0; }

double gelu(double x) {
    const double c = std::sqrt(2.0 / M_PI);
    return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
}

// The value of `op` applied to splat operands, for the ops whose result is again a splat.
std::optional<double> foldSplat(const Operation& op, const std::vector<double>& in) {
    const std::string& name = op.getOpName();
    if (name == "add") return in[0] + in[1];
    if (name == "sub") return in[0] - in[1];
    if (name == "mul") return in[0] * in[1];
    if (name == "div") return in[0] / in[1];
    if (name == "relu") return relu(in[0]);
    if (name == "gelu") return gelu(in[0]);
    if (name == "exp") return std::exp(in[0]);
    if (name == "log") return std::log(in[0]);
    if (name == "tanh") return std::tanh(in[0]);
    if (name == "sqrt") return std::sqrt(in[0]);
    if (name == "rsqrt") return 1.0 / std::sqrt(in[0]);
    if (name == "cmp_gt") return in[0] > in[1] ? 1.0 : 0.0;
    if (name == "select") return in[0] != 0.0 ? in[1] : in[2];
    if (name == "broadcast_to" || name == "reshape" || name == "transpose") return in[0];
    if (name == "sum" || name == "mean" || name == "reduce_max") {
        const Shape& shape = static_cast<const TensorType&>(*op.getOperand(0)->getType()).getShape();
        int64_t count = 1;
        for (int64_t axis : op.getIntListAttr("axes")) count *= shape[axis < 0 ? axis + shape.size() : axis];
        if (name == "sum") return count == 0 ? 0.0 : in[0] * static_cast<double>(count);
        return count == 0 ? std::numeric_limits<double>::quiet_NaN() : in[0];
    }
    return std::nullopt;
}

class Rewriter {
public:
    explicit Rewriter(Function& func) : func_(func) {}

    bool run() {
        bool changedAny = false;
        for (int iteration = 0; iteration < 64; ++iteration) {
            numberValues();
            bool changed = false;
            const std::vector<OperationPtr> ops = func_.getEntryBlock()->getOperations();
            for (const auto& op : ops) {
                if (op->getParentBlock() && op->getNumResults() == 1) changed |= rewrite(*op);
            }
            if (!changed) break;
            changedAny = true;
        }
        return changedAny;
    }

private:
    // Replaces `op`'s result with `value` and erases `op`.
    bool replace(Operation& op, const ValuePtr& value) {
        replaceAllUsesWith(op.getResult(0), value);
        op.getParentBlock()->eraseOperation(&op);
        return true;
    }

    // `value` in place of `op`'s result, broadcast first if the result is larger.
    bool replaceKeepingType(Operation& op, const ValuePtr& value) {
        if (sameType(value, op.getResult(0))) return replace(op, value);
        if (tensor(value).getElementType() != tensor(op.getResult(0)).getElementType()) return false;
        IRBuilder b;
        b.setInsertPointBefore(&op);
        return replace(op, b.createBroadcastTo(value, tensor(op.getResult(0)).getShape()));
    }

    bool rewrite(Operation& op) {
        const std::string& name = op.getOpName();
        if (name == "constant") return false;

        // Constant folding: every operand is a splat, and so is the result.
        std::vector<double> values;
        for (const auto& operand : op.getOperands()) {
            if (auto v = splat(operand)) values.push_back(*v);
        }
        if (!values.empty() && values.size() == op.getNumOperands() &&
            tensor(op.getResult(0)).getElementType() == DataType::Float32) {
            if (auto folded = foldSplat(op, values)) {
                IRBuilder b;
                b.setInsertPointBefore(&op);
                return replace(op, b.createConstant(*folded, op.getResult(0)->getType()));
            }
        }

        if (name == "add" || name == "sub" || name == "mul" || name == "div") return arithmetic(op);
        if (name == "select") {
            if (auto c = splat(op.getOperand(0))) return replaceKeepingType(op, op.getOperand(*c != 0.0 ? 1 : 2));
            return false;
        }
        if (name == "matmul") return foldTransposes(op);
        if (name == "transpose") return transpose(op);
        if (name == "reshape" || name == "broadcast_to") {
            ValuePtr in = op.getOperand(0);
            if (sameType(in, op.getResult(0))) return replace(op, in);
            const Operation* def = in->getDefiningOp();
            if (def && def->getOpName() == name) { // reshape(reshape(x)) / broadcast_to(broadcast_to(x))
                op.setOperand(0, def->getOperand(0));
                return true;
            }
        }
        return false;
    }

    bool arithmetic(Operation& op) {
        const std::string& name = op.getOpName();
        ValuePtr lhs = op.getOperand(0);
        ValuePtr rhs = op.getOperand(1);
        auto l = splat(lhs);
        auto r = splat(rhs);
        if (name == "add" && r && *r == 0.0) return replaceKeepingType(op, lhs);
        if (name == "add" && l && *l == 0.0) return replaceKeepingType(op, rhs);
        if (name == "sub" && r && *r == 0.0) return replaceKeepingType(op, lhs);
        if (name == "mul" && r && *r == 1.0) return replaceKeepingType(op, lhs);
        if (name == "mul" && l && *l == 1.0) return replaceKeepingType(op, rhs);
        if (name == "div" && r && *r == 1.0) return replaceKeepingType(op, lhs);
        // Commutative ops: constants second, otherwise the earlier definition first.
        if ((name == "add" || name == "mul") && rank(lhs) > rank(rhs)) {
            op.setOperand(0, rhs);
            op.setOperand(1, lhs);
            return true;
        }
        return false;
    }

    // matmul(transpose(A) swapping the last two axes, B) → matmul(A, B) with transpose_a toggled; same for B.
    bool foldTransposes(Operation& op) {
        bool changed = false;
        for (size_t i = 0; i < 2; ++i) {
            const Operation* def = op.getOperand(i)->getDefiningOp();
            if (!def || def->getOpName() != "transpose") continue;
            const auto& perm = def->getIntListAttr("perm");
            size_t rank = perm.size();
            bool swapsLastTwo = rank >= 2 && perm[rank - 2] == static_cast<int64_t>(rank - 1) &&
                                perm[rank - 1] == static_cast<int64_t>(rank - 2);
            for (size_t k = 0; swapsLastTwo && k + 2 < rank; ++k) swapsLastTwo = perm[k] == static_cast<int64_t>(k);
            if (!swapsLastTwo) continue;
            const char* flagName = i == 0 ? "transpose_a" : "transpose_b";
            bool set = op.hasAttribute(flagName) && op.getIntAttr(flagName) == 1;
            if (set) op.removeAttribute(flagName);
            else op.setAttribute(flagName, int64_t{1});
            op.setOperand(i, def->getOperand(0));
            changed = true;
        }
        return changed;
    }

    bool transpose(Operation& op) {
        const auto& perm = op.getIntListAttr("perm");
        bool identity = true;
        for (size_t i = 0; i < perm.size(); ++i) identity = identity && perm[i] == static_cast<int64_t>(i);
        if (identity) return replace(op, op.getOperand(0));
        const Operation* def = op.getOperand(0)->getDefiningOp();
        if (!def || def->getOpName() != "transpose") return false;
        // transpose(transpose(x, p), q) = transpose(x, r) with r[i] = p[q[i]]
        const auto& inner = def->getIntListAttr("perm");
        std::vector<int64_t> composed(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) composed[i] = inner[perm[i]];
        op.setAttribute("perm", composed);
        op.setOperand(0, def->getOperand(0));
        return true;
    }

    // Order key for commutative operands: arguments, then op results in program order, constants last.
    int64_t rank(const ValuePtr& value) const {
        if (splat(value)) return std::numeric_limits<int64_t>::max();
        auto it = numbers_.find(value.get());
        return it == numbers_.end() ? -1 : it->second;
    }

    void numberValues() {
        numbers_.clear();
        int64_t next = 0;
        for (const auto& arg : func_.getArguments()) numbers_[arg.get()] = next++;
        for (const auto& op : func_.getEntryBlock()->getOperations()) {
            for (const auto& result : op->getResults()) numbers_[result.get()] = next++;
        }
    }

    Function& func_;
    std::unordered_map<const Value*, int64_t> numbers_;
};

} // namespace

bool CanonicalizePass::runOnFunction(Function& func) { return Rewriter(func).run(); }

} // namespace dlc
