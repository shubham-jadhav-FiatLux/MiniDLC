#include "dlc/passes/Autodiff.h"
#include "dlc/ir/Builder.h"
#include "dlc/ir/Error.h"
#include "dlc/passes/Activity.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <unordered_map>

namespace dlc {

namespace {

const Shape& shapeOf(const ValuePtr& value) { return static_cast<const TensorType&>(*value->getType()).getShape(); }

int64_t product(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

bool flagSet(const Operation& op, const char* name) { return op.hasAttribute(name) && op.getIntAttr(name) == 1; }

std::vector<int64_t> reducedAxes(const Operation& op, size_t rank) {
    std::vector<int64_t> axes;
    for (int64_t a : op.getIntListAttr("axes")) axes.push_back(a < 0 ? a + static_cast<int64_t>(rank) : a);
    std::sort(axes.begin(), axes.end());
    return axes;
}

using Accumulate = std::function<void(const ValuePtr&, ValuePtr)>;

// Builds the vector-Jacobian products of one op. Result types are inferred, so a rule that builds an ill-typed op
// fails with an error instead of producing invalid IR.
class VJPBuilder {
public:
    VJPBuilder(IRBuilder& b, const ActivityAnalysis& activity, Accumulate accumulate)
        : b_(b), activity_(activity), accumulate_(std::move(accumulate)) {}

    void apply(Operation& op, const ValuePtr& dY);

private:
    ValuePtr scalar(double value) { return b_.createScalar(value); }
    ValuePtr neg(const ValuePtr& x) { return b_.createMul(x, scalar(-1.0)); }

    // Sums dY over the dimensions that broadcasting added on the way from `target` to dY's shape.
    ValuePtr unbroadcast(const ValuePtr& dY, const Shape& target) {
        const Shape& ys = shapeOf(dY);
        if (ys == target) return dY;
        size_t extra = ys.size() - target.size();
        std::vector<int64_t> axes;
        for (size_t i = 0; i < extra; ++i) axes.push_back(static_cast<int64_t>(i));
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i] == 1 && ys[extra + i] != 1) axes.push_back(static_cast<int64_t>(extra + i));
        }
        ValuePtr r = axes.empty() ? dY : b_.createReduce("sum", dY, axes, /*keepdims=*/true);
        return shapeOf(r) == target ? r : b_.createReshape(r, target);
    }

    // Broadcasts the gradient (or value) of a reduction back over the reduced axes.
    ValuePtr expand(ValuePtr dY, const Shape& inShape, const std::vector<int64_t>& axes, bool keepdims) {
        if (!keepdims) {
            Shape kept = inShape;
            for (int64_t a : axes) kept[a] = 1;
            if (shapeOf(dY) != kept) dY = b_.createReshape(dY, kept);
        }
        return shapeOf(dY) == inShape ? dY : b_.createBroadcastTo(dY, inShape);
    }

    void matmul(Operation& op, const ValuePtr& dY);
    void layernorm(Operation& op, const ValuePtr& dY);

    bool needs(const Operation& op, size_t idx) const {
        return ActivityAnalysis::isDifferentiableOperand(op, idx) && activity_.isActive(op.getOperand(idx).get());
    }
    void add(const ValuePtr& target, ValuePtr grad) { accumulate_(target, std::move(grad)); }

    IRBuilder& b_;
    const ActivityAnalysis& activity_;
    Accumulate accumulate_;
};

void VJPBuilder::apply(Operation& op, const ValuePtr& dY) {
    const std::string& name = op.getOpName();
    auto x = [&op](size_t i) { return op.getOperand(i); };
    const ValuePtr Y = op.getResult(0);

    if (name == "add" || name == "sub") {
        if (needs(op, 0)) add(x(0), unbroadcast(dY, shapeOf(x(0))));
        if (needs(op, 1)) add(x(1), unbroadcast(name == "add" ? dY : neg(dY), shapeOf(x(1))));
    } else if (name == "mul") {
        if (needs(op, 0)) add(x(0), unbroadcast(b_.createMul(dY, x(1)), shapeOf(x(0))));
        if (needs(op, 1)) add(x(1), unbroadcast(b_.createMul(dY, x(0)), shapeOf(x(1))));
    } else if (name == "div") {
        // y = a / b: da = dy / b, db = -dy * y / b
        if (needs(op, 0)) add(x(0), unbroadcast(b_.createDiv(dY, x(1)), shapeOf(x(0))));
        if (needs(op, 1)) add(x(1), unbroadcast(neg(b_.createMul(dY, b_.createDiv(Y, x(1)))), shapeOf(x(1))));
    } else if (name == "relu") {
        if (needs(op, 0)) add(x(0), b_.createSelect(b_.createBinary("cmp_gt", x(0), scalar(0.0)), dY, scalar(0.0)));
    } else if (name == "gelu") {
        // y = x/2 (1 + t), t = tanh(c (x + k x^3)): dy/dx = (1 + t)/2 + x/2 (1 - t^2) c (1 + 3k x^2)
        if (needs(op, 0)) {
            const double c = std::sqrt(2.0 / M_PI);
            const double k = 0.044715;
            ValuePtr in = x(0);
            ValuePtr x2 = b_.createMul(in, in);
            ValuePtr t = b_.createUnary("tanh", b_.createMul(b_.createAdd(in, b_.createMul(b_.createMul(x2, in),
                                                                                           scalar(k))),
                                                             scalar(c)));
            ValuePtr left = b_.createMul(b_.createAdd(t, scalar(1.0)), scalar(0.5));
            ValuePtr oneMinusT2 = b_.createSub(scalar(1.0), b_.createMul(t, t));
            ValuePtr poly = b_.createAdd(b_.createMul(x2, scalar(3.0 * k)), scalar(1.0));
            ValuePtr right = b_.createMul(b_.createMul(b_.createMul(in, scalar(0.5 * c)), oneMinusT2), poly);
            add(in, b_.createMul(dY, b_.createAdd(left, right)));
        }
    } else if (name == "exp") {
        if (needs(op, 0)) add(x(0), b_.createMul(dY, Y));
    } else if (name == "log") {
        if (needs(op, 0)) add(x(0), b_.createDiv(dY, x(0)));
    } else if (name == "tanh") {
        if (needs(op, 0)) add(x(0), b_.createMul(dY, b_.createSub(scalar(1.0), b_.createMul(Y, Y))));
    } else if (name == "sqrt") {
        if (needs(op, 0)) add(x(0), b_.createDiv(b_.createMul(dY, scalar(0.5)), Y));
    } else if (name == "rsqrt") {
        // y = x^(-1/2): dy/dx = -y^3 / 2
        if (needs(op, 0)) add(x(0), b_.createMul(dY, b_.createMul(b_.createMul(b_.createMul(Y, Y), Y), scalar(-0.5))));
    } else if (name == "matmul") {
        matmul(op, dY);
    } else if (name == "softmax") {
        // dx = y * (dy - sum(dy * y))
        int64_t axis = op.getIntAttr("axis");
        if (needs(op, 0)) {
            ValuePtr dot = b_.createReduce("sum", b_.createMul(dY, Y), {axis}, /*keepdims=*/true);
            add(x(0), b_.createMul(Y, b_.createSub(dY, dot)));
        }
    } else if (name == "log_softmax") {
        // dx = dy - softmax(x) * sum(dy), with softmax(x) = exp(y)
        int64_t axis = op.getIntAttr("axis");
        if (needs(op, 0)) {
            ValuePtr total = b_.createReduce("sum", dY, {axis}, /*keepdims=*/true);
            add(x(0), b_.createSub(dY, b_.createMul(b_.createUnary("exp", Y), total)));
        }
    } else if (name == "cross_entropy") {
        // dz = (softmax(z) - onehot(y)) / N, N = number of rows
        if (needs(op, 0)) {
            const Shape zs = shapeOf(x(0));
            int64_t classes = zs.back();
            int64_t rows = classes == 0 ? 0 : product(zs) / classes;
            ValuePtr diff = b_.createSub(b_.createSoftmax(x(0), -1), b_.createOneHot(x(1), classes));
            add(x(0), b_.createMul(diff, b_.createMul(dY, scalar(rows == 0 ? 0.0 : 1.0 / rows))));
        }
    } else if (name == "layernorm") {
        layernorm(op, dY);
    } else if (name == "sum" || name == "mean" || name == "reduce_max") {
        if (needs(op, 0)) {
            const Shape xs = shapeOf(x(0));
            std::vector<int64_t> axes = reducedAxes(op, xs.size());
            bool keepdims = op.getIntAttr("keepdims") != 0;
            ValuePtr grad = expand(dY, xs, axes, keepdims);
            if (name == "mean") {
                int64_t n = 1;
                for (int64_t a : axes) n *= xs[a];
                grad = b_.createMul(grad, scalar(n == 0 ? 0.0 : 1.0 / static_cast<double>(n)));
            } else if (name == "reduce_max") {
                // Ties share the gradient equally.
                ValuePtr mask = b_.createSelect(b_.createBinary("cmp_gt", expand(Y, xs, axes, keepdims), x(0)),
                                                scalar(0.0), scalar(1.0));
                grad = b_.createDiv(b_.createMul(grad, mask), b_.createReduce("sum", mask, axes, /*keepdims=*/true));
            }
            add(x(0), grad);
        }
    } else if (name == "transpose") {
        if (needs(op, 0)) {
            const auto& perm = op.getIntListAttr("perm");
            std::vector<int64_t> inverse(perm.size());
            for (size_t i = 0; i < perm.size(); ++i) inverse[perm[i]] = static_cast<int64_t>(i);
            add(x(0), b_.createTranspose(dY, inverse));
        }
    } else if (name == "reshape") {
        if (needs(op, 0)) add(x(0), b_.createReshape(dY, shapeOf(x(0))));
    } else if (name == "broadcast_to") {
        if (needs(op, 0)) add(x(0), unbroadcast(dY, shapeOf(x(0))));
    } else if (name == "embedding") {
        if (needs(op, 0)) add(x(0), b_.createScatterAdd(x(1), dY, shapeOf(x(0))[0]));
    } else if (name == "scatter_add") {
        if (needs(op, 1)) add(x(1), b_.createEmbedding(dY, x(0)));
    } else if (name == "select") {
        if (needs(op, 1)) add(x(1), unbroadcast(b_.createSelect(x(0), dY, scalar(0.0)), shapeOf(x(1))));
        if (needs(op, 2)) add(x(2), unbroadcast(b_.createSelect(x(0), scalar(0.0), dY), shapeOf(x(2))));
    } else {
        throw IRError("autodiff: no VJP rule for '" + name + "'");
    }
}

// y = op(A) op(B), where op transposes the last two dimensions when the flag is set. Same-rank operands (2-D or
// batched): dA = dY op(B)^T (transposed again if A was), dB = op(A)^T dY likewise. A 3-D x 2-D matmul shares B
// across the batch, so dB sums over it: both are flattened to 2-D first.
void VJPBuilder::matmul(Operation& op, const ValuePtr& dY) {
    ValuePtr a = op.getOperand(0);
    ValuePtr b = op.getOperand(1);
    bool ta = flagSet(op, "transpose_a");
    bool tb = flagSet(op, "transpose_b");
    size_t ra = shapeOf(a).size();
    size_t rb = shapeOf(b).size();
    if (ra == rb) {
        if (needs(op, 0)) add(a, ta ? b_.createMatmul(b, dY, tb, true) : b_.createMatmul(dY, b, false, !tb));
        if (needs(op, 1)) add(b, tb ? b_.createMatmul(dY, a, true, ta) : b_.createMatmul(a, dY, !ta, false));
        return;
    }
    if (ra != 3 || rb != 2) throw IRError("autodiff: no VJP rule for this matmul operand ranks");
    if (needs(op, 0)) {
        ValuePtr dAhat = b_.createMatmul(dY, b, false, !tb);
        add(a, ta ? b_.createTranspose(dAhat, {0, 2, 1}) : dAhat);
    }
    if (needs(op, 1)) {
        ValuePtr aHat = ta ? b_.createTranspose(a, {0, 2, 1}) : a;
        const Shape as = shapeOf(aHat);
        const Shape ys = shapeOf(dY);
        ValuePtr a2 = b_.createReshape(aHat, {as[0] * as[1], as[2]});
        ValuePtr dY2 = b_.createReshape(dY, {ys[0] * ys[1], ys[2]});
        add(b, tb ? b_.createMatmul(dY2, a2, true, false) : b_.createMatmul(a2, dY2, true, false));
    }
}

// x̂ = (x - μ) rstd, rstd = 1/√(σ² + ε), y = γ x̂ + β, all over the last axis. dβ = Σ dy, dγ = Σ dy x̂ (over the
// leading axes); with g = dy γ: dx = (rstd / D) (D g - Σ g - x̂ Σ g x̂) (over the last axis).
void VJPBuilder::layernorm(Operation& op, const ValuePtr& dY) {
    ValuePtr x = op.getOperand(0);
    ValuePtr gamma = op.getOperand(1);
    ValuePtr beta = op.getOperand(2);
    const Shape& xs = shapeOf(x);
    int64_t last = static_cast<int64_t>(xs.size()) - 1;
    std::vector<int64_t> lead(xs.size() - 1);
    std::iota(lead.begin(), lead.end(), 0);
    auto sumLead = [&](const ValuePtr& v) { return lead.empty() ? v : b_.createReduce("sum", v, lead, false); };

    if (needs(op, 2)) add(beta, sumLead(dY));
    if (!needs(op, 0) && !needs(op, 1)) return;
    ValuePtr mu = b_.createReduce("mean", x, {last}, /*keepdims=*/true);
    ValuePtr centered = b_.createSub(x, mu);
    ValuePtr var = b_.createReduce("mean", b_.createMul(centered, centered), {last}, /*keepdims=*/true);
    ValuePtr rstd = b_.createUnary("rsqrt", b_.createAdd(var, scalar(op.getDoubleAttr("eps"))));
    ValuePtr xhat = b_.createMul(centered, rstd);
    if (needs(op, 1)) add(gamma, sumLead(b_.createMul(dY, xhat)));
    if (needs(op, 0)) {
        double d = static_cast<double>(xs.back());
        ValuePtr g = b_.createMul(dY, gamma);
        ValuePtr sumG = b_.createReduce("sum", g, {last}, /*keepdims=*/true);
        ValuePtr sumGX = b_.createReduce("sum", b_.createMul(g, xhat), {last}, /*keepdims=*/true);
        ValuePtr inner = b_.createSub(b_.createSub(b_.createMul(g, scalar(d)), sumG), b_.createMul(xhat, sumGX));
        add(x, b_.createMul(b_.createMul(rstd, scalar(d == 0 ? 0.0 : 1.0 / d)), inner));
    }
}

} // namespace

bool AutodiffPass::runOnFunction(Function& func) {
    Block* entry = func.getEntryBlock();
    const auto& ops = entry->getOperations();
    if (ops.empty() || ops.back()->getOpName() != "return" || ops.back()->getNumOperands() == 0) {
        throw IRError("autodiff: the function must return its loss as the first result");
    }
    for (const auto& info : func.getResultInfo()) {
        if (info.kind != ResultKind::Value) throw IRError("autodiff: the function's results are already annotated");
    }
    Operation* ret = ops.back().get();
    ValuePtr loss = ret->getOperand(0);
    if (loss->getType()->toString() != "f32[]") {
        throw IRError("autodiff: the loss (the first result) must be f32[], got " + loss->getType()->toString());
    }
    const auto& args = func.getArguments();
    for (size_t i = 0; i < args.size(); ++i) {
        if (func.getArgRole(i) == ArgRole::Param && args[i]->getType()->toString().rfind("f32", 0) != 0) {
            throw IRError("autodiff: param argument " + std::to_string(i) + " must be f32");
        }
    }

    ActivityAnalysis activity(func, loss.get());
    std::vector<ValuePtr> results = ret->getOperands();
    const std::vector<OperationPtr> forward(ops.begin(), ops.end() - 1);
    entry->eraseOperation(ret);

    IRBuilder builder(entry);
    std::unordered_map<const Value*, ValuePtr> adjoints;
    adjoints[loss.get()] = builder.createScalar(1.0);
    VJPBuilder vjp(builder, activity, [&](const ValuePtr& target, ValuePtr grad) {
        if (!activity.isActive(target.get())) return;
        auto [it, inserted] = adjoints.emplace(target.get(), grad);
        if (!inserted) it->second = builder.createAdd(it->second, grad);
    });
    for (auto it = forward.rbegin(); it != forward.rend(); ++it) {
        Operation& op = **it;
        if (op.getNumResults() == 0) continue;
        auto adjoint = adjoints.find(op.getResult(0).get());
        if (adjoint != adjoints.end() && activity.isActive(op.getResult(0).get())) vjp.apply(op, adjoint->second);
    }

    std::vector<TypePtr> types = func.getResultTypes();
    std::vector<ResultInfo> info = func.getResultInfo();
    for (size_t i = 0; i < args.size(); ++i) {
        if (func.getArgRole(i) != ArgRole::Param) continue;
        auto it = adjoints.find(args[i].get());
        // A parameter the loss does not depend on gets a zero gradient, so every parameter has one.
        results.push_back(it != adjoints.end() ? it->second : builder.createConstant(0.0, args[i]->getType()));
        types.push_back(args[i]->getType());
        info.push_back({ResultKind::Grad, i});
    }
    builder.createReturn(results);
    func.setResults(types, info);
    return true;
}

} // namespace dlc
