#include "dlc/loopir/Lowering.h"
#include "dlc/loopir/Analysis.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/SlotTracker.h"
#include "dlc/passes/CostModel.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace dlc::loopir {

std::string targetName(Target target) { return target == Target::CPU ? "cpu" : "gpu"; }

namespace {

// Kernels with less work than this run on one thread: below it, dispatching to the pool costs more than it saves.
constexpr int64_t kParallelWork = 1 << 15;
// Elements per parallel chunk of a flat elementwise loop.
constexpr int64_t kChunk = 8192;
// The CPU matmul register tile: MR rows by NR columns of accumulators (8 AVX2 registers of 8 floats).
constexpr int64_t kMR = 4;
constexpr int64_t kNR = 16;
constexpr int kVector = 8;

const Shape& shapeOf(const Value* v) { return static_cast<const TensorType&>(*v->getType()).getShape(); }

int64_t numel(const Shape& s) { return std::accumulate(s.begin(), s.end(), int64_t{1}, std::multiplies<int64_t>()); }

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

ScalarType typeOf(const Value* v) {
    return elementType(static_cast<const TensorType&>(*v->getType()).getElementType());
}

std::optional<double> splatValue(const Value* v) {
    const Operation* def = v->getDefiningOp();
    if (!def || def->getOpName() != "constant") return std::nullopt;
    return def->getDoubleAttr("value");
}

// Builds one loop kernel: a global buffer per planned buffer it touches, fresh names for its variables.
class KernelBuilder {
public:
    KernelBuilder(std::string name, const BufferAssignment& buffers, const SlotTracker& names)
        : buffers_(buffers), names_(names) {
        out_.kernel.name = std::move(name);
    }

    // Registers the kernel's inputs, then its outputs, as global buffers: the parameter order is deterministic.
    void declare(const dlc::Kernel& k) {
        for (const Value* in : k.inputs) param(in); // fusion leaves inlined splats out of the inputs
        for (const Value* out : k.outputs) param(out);
    }

    int param(const Value* v) {
        const Buffer* b = &buffers_.bufferOf(v);
        auto it = params_.find(b);
        if (it != params_.end()) return it->second;
        int id = out_.kernel.addBuffer({"%" + names_.name(b->value), typeOf(v), shapeOf(b->value), MemScope::Global});
        out_.args.push_back(b);
        params_[b] = id;
        return id;
    }

    int local(const std::string& base, ScalarType type, Shape shape, MemScope scope) {
        return out_.kernel.addBuffer({fresh(base), type, std::move(shape), scope});
    }

    ExprPtr loadValue(const Value* v, ExprPtr index) { return load(param(v), std::move(index), typeOf(v)); }

    std::string fresh(const std::string& base) { return base + std::to_string(counter_++); }

    LoweredKernel finish(Body body, const dlc::Kernel& source, bool first) {
        out_.kernel.body = std::move(body);
        out_.graphKernel = source.name;
        if (first) {
            Cost c = kernelCost(source);
            out_.kernel.flops = c.flops;
            out_.kernel.bytes = c.bytes();
        }
        return std::move(out_);
    }

private:
    const BufferAssignment& buffers_;
    const SlotTracker& names_;
    LoweredKernel out_;
    std::unordered_map<const Buffer*, int> params_;
    int counter_{0};
};

// A Let whose value is then referred to by name.
ExprPtr bindLocal(KernelBuilder& kb, Body& body, const std::string& base, ExprPtr value) {
    if (value->kind == ExprKind::Int || value->kind == ExprKind::Float || value->kind == ExprKind::Var) return value;
    std::string name = kb.fresh(base);
    ScalarType type = value->type;
    body.push_back(let(name, type, std::move(value)));
    return var(name, type);
}

// `if (0 <= id < n) {then} else {otherwise}`: indices read from memory are checked, so any input is memory-safe.
StmtPtr guardIndex(const std::string& id, int64_t n, Body then, Body otherwise = {}) {
    Body inner = {ifThen(compare(CompareOp::LT, var(id), intConst(n)), then, otherwise)};
    return ifThen(compare(CompareOp::GE, var(id), intConst(0)), std::move(inner), otherwise);
}

ExprPtr pointwise(const Operation& op, const std::vector<ExprPtr>& in) {
    const std::string& name = op.getOpName();
    auto f = [](double v) { return floatConst(v); };
    if (name == "add") return add(in[0], in[1]);
    if (name == "sub") return sub(in[0], in[1]);
    if (name == "mul") return mul(in[0], in[1]);
    if (name == "div") return div(in[0], in[1]);
    if (name == "relu") return binary(BinaryOp::MaxNaN, in[0], f(0.0));
    if (name == "gelu") {
        // 0.5 x (1 + tanh(c (x + 0.044715 x^3)))
        const double c = std::sqrt(2.0 / M_PI);
        ExprPtr x3 = mul(mul(in[0], in[0]), in[0]);
        ExprPtr t = unary(UnaryOp::Tanh, mul(f(c), add(in[0], mul(f(0.044715), x3))));
        return mul(mul(f(0.5), in[0]), add(f(1.0), t));
    }
    if (name == "exp") return unary(UnaryOp::Exp, in[0]);
    if (name == "log") return unary(UnaryOp::Log, in[0]);
    if (name == "tanh") return unary(UnaryOp::Tanh, in[0]);
    if (name == "sqrt") return unary(UnaryOp::Sqrt, in[0]);
    if (name == "rsqrt") return unary(UnaryOp::Rsqrt, in[0]);
    if (name == "cmp_gt") return select(compare(CompareOp::GT, in[0], in[1]), f(1.0), f(0.0));
    if (name == "select") return select(compare(CompareOp::NE, in[0], f(0.0)), in[1], in[2]);
    if (name == "cast") return cast(op.getStringAttr("to") == "i32" ? ScalarType::I32 : ScalarType::F32, in[0]);
    if (name == "broadcast_to") return in[0];
    throw IRError("lowering: '" + name + "' is not elementwise");
}

// Evaluates the members of a fused group at one output position: values inside the group come from earlier members,
// splat constants are immediates, and the rest are loaded with broadcasting. `index(v)` is the flat index of v at the
// position. Loads all happen before the stores, so a kernel may overwrite an input it reads (an in-place update).
class PointEmitter {
public:
    PointEmitter(KernelBuilder& kb, Body& body, std::function<ExprPtr(const Value*)> index)
        : kb_(kb), body_(body), index_(std::move(index)) {}

    void bind(const Value* v, ExprPtr value) { values_[v] = std::move(value); }

    ExprPtr value(const Value* v) {
        auto it = values_.find(v);
        if (it != values_.end()) return it->second;
        ExprPtr e;
        if (auto s = splatValue(v)) {
            e = typeOf(v) == ScalarType::F32 ? floatConst(*s) : cast(typeOf(v), intConst(static_cast<int64_t>(*s)));
        } else {
            e = bindLocal(kb_, body_, "in", kb_.loadValue(v, index_(v)));
        }
        values_[v] = e;
        return e;
    }

    void member(const Operation& op) {
        std::vector<ExprPtr> in;
        for (const auto& operand : op.getOperands()) in.push_back(value(operand.get()));
        values_[op.getResult(0).get()] = bindLocal(kb_, body_, "v", pointwise(op, in));
    }

    void storeOutputs(const std::vector<Value*>& outputs) {
        for (const Value* out : outputs) body_.push_back(store(kb_.param(out), index_(out), value(out)));
    }

private:
    KernelBuilder& kb_;
    Body& body_;
    std::function<ExprPtr(const Value*)> index_;
    std::unordered_map<const Value*, ExprPtr> values_;
};

// Row-major strides.
std::vector<int64_t> stridesOf(const Shape& s) {
    std::vector<int64_t> st(s.size(), 1);
    for (int64_t d = static_cast<int64_t>(s.size()) - 2; d >= 0; --d) st[d] = st[d + 1] * s[d + 1];
    return st;
}

// The flat index of `v` (broadcast against `outShape`) at the output position `idx` (one expression per output dim).
ExprPtr broadcastIndex(const Value* v, const Shape& outShape, const std::vector<ExprPtr>& idx) {
    const Shape& s = shapeOf(v);
    auto st = stridesOf(s);
    size_t offset = outShape.size() - s.size();
    ExprPtr index = intConst(0);
    for (size_t d = 0; d < s.size(); ++d) {
        if (s[d] == 1) continue;
        index = add(index, mul(idx[offset + d], intConst(st[d])));
    }
    return index;
}

// The multi-index of a row-major flat index `flat` into `shape`.
std::vector<ExprPtr> unflatten(const ExprPtr& flat, const Shape& shape) {
    std::vector<ExprPtr> idx(shape.size());
    auto st = stridesOf(shape);
    for (size_t d = 0; d < shape.size(); ++d) {
        ExprPtr q = shape[d] == 1 ? intConst(0) : div(flat, intConst(st[d]));
        idx[d] = d == 0 ? q : rem(q, intConst(shape[d]));
        if (shape[d] == 1) idx[d] = intConst(0);
    }
    return idx;
}

// --- Elementwise -------------------------------------------------------------------------------------------------

// The iteration space after dropping size-1 dimensions and merging neighbours that every accessed value either
// spans fully or broadcasts over: fewer, longer loops, and a flat one when no operand broadcasts.
struct Collapsed {
    std::vector<int64_t> extents;
    // Per value: its stride for each collapsed dimension (0 where it broadcasts).
    std::unordered_map<const Value*, std::vector<int64_t>> strides;
};

Collapsed collapse(const Shape& out, const std::vector<const Value*>& values) {
    std::vector<size_t> dims;
    for (size_t d = 0; d < out.size(); ++d) {
        if (out[d] != 1) dims.push_back(d);
    }
    auto full = [&](const Value* v, size_t d) {
        const Shape& s = shapeOf(v);
        size_t offset = out.size() - s.size();
        return d >= offset && s[d - offset] == out[d];
    };
    std::vector<std::vector<size_t>> groups;
    for (size_t d : dims) {
        bool merge = !groups.empty();
        for (const Value* v : values) merge = merge && full(v, groups.back().back()) == full(v, d);
        if (merge) groups.back().push_back(d);
        else groups.push_back({d});
    }
    Collapsed c;
    for (const auto& g : groups) {
        int64_t e = 1;
        for (size_t d : g) e *= out[d];
        c.extents.push_back(e);
    }
    for (const Value* v : values) {
        std::vector<int64_t> st(groups.size(), 0);
        int64_t stride = 1;
        for (int64_t gi = static_cast<int64_t>(groups.size()) - 1; gi >= 0; --gi) {
            if (!full(v, groups[gi][0])) continue;
            st[gi] = stride;
            stride *= c.extents[gi];
        }
        c.strides[v] = st;
    }
    return c;
}

std::vector<LoweredKernel> lowerElementwise(const dlc::Kernel& k, const BufferAssignment& ba, const SlotTracker& names) {
    KernelBuilder kb(k.name, ba, names);
    kb.declare(k);
    std::vector<const Value*> accessed;
    std::unordered_set<const Operation*> members(k.ops.begin(), k.ops.end());
    for (const Value* in : k.inputs) accessed.push_back(in);
    for (const Value* out : k.outputs) accessed.push_back(out);
    Collapsed c = collapse(k.iterationShape, accessed);
    const int64_t total = numel(k.iterationShape);
    const bool parallel = total * static_cast<int64_t>(k.ops.size()) >= kParallelWork;

    // The body at one position, given one index expression per collapsed dimension.
    auto point = [&](const std::vector<ExprPtr>& idx) {
        Body body;
        PointEmitter pe(kb, body, [&](const Value* v) {
            ExprPtr index = intConst(0);
            const auto& st = c.strides.at(v);
            for (size_t d = 0; d < idx.size(); ++d) {
                if (st[d] != 0) index = add(index, mul(idx[d], intConst(st[d])));
            }
            return index;
        });
        for (const Operation* op : k.ops) pe.member(*op);
        pe.storeOutputs(k.outputs);
        return body;
    };

    Body top;
    const size_t rank = c.extents.size();
    if (rank == 0) {
        top = point({});
    } else if (rank == 1) {
        int64_t n = c.extents[0];
        if (parallel && n > kChunk) {
            int64_t chunks = ceilDiv(n, kChunk);
            int64_t tail = n - (chunks - 1) * kChunk;
            auto chunk = [&](int64_t extent) {
                Body inner = point({add(mul(var("c"), intConst(kChunk)), var("e"))});
                return forLoop("e", extent, LoopKind::Vectorized, std::move(inner), kVector);
            };
            Body chunkBody;
            if (tail == kChunk) {
                chunkBody.push_back(chunk(kChunk));
            } else {
                StmtPtr full = chunk(kChunk); // built in order, so variable names are deterministic
                StmtPtr last = chunk(tail);
                chunkBody.push_back(ifThen(compare(CompareOp::LT, var("c"), intConst(chunks - 1)), {full}, {last}));
            }
            top.push_back(forLoop("c", chunks, LoopKind::Parallel, std::move(chunkBody)));
        } else {
            top.push_back(forLoop("e", n, LoopKind::Vectorized, point({var("e")}), kVector));
        }
    } else {
        std::vector<ExprPtr> idx;
        for (size_t d = 0; d < rank; ++d) idx.push_back(var("d" + std::to_string(d)));
        Body inner = point(idx);
        StmtPtr loop = forLoop("d" + std::to_string(rank - 1), c.extents[rank - 1], LoopKind::Vectorized,
                               std::move(inner), kVector);
        for (int64_t d = static_cast<int64_t>(rank) - 2; d >= 0; --d) {
            LoopKind kind = d == 0 && parallel ? LoopKind::Parallel : LoopKind::Serial;
            loop = forLoop("d" + std::to_string(d), c.extents[d], kind, {loop});
        }
        top.push_back(loop);
    }
    std::vector<LoweredKernel> out;
    out.push_back(kb.finish(std::move(top), k, true));
    return out;
}

// --- Matmul --------------------------------------------------------------------------------------------------------

struct MatmulShape {
    int64_t batch{1}, m{0}, n{0}, k{0};
    int64_t aBatchStride{0}, bBatchStride{0};
    bool ta{false}, tb{false};
    bool flattened{false}; // a 3-D x 2-D matmul run as one 2-D matmul over B*M rows
    Shape outShape;
};

MatmulShape matmulShape(const Operation& op) {
    MatmulShape s;
    const Shape& a = shapeOf(op.getOperand(0).get());
    const Shape& b = shapeOf(op.getOperand(1).get());
    s.outShape = shapeOf(op.getResult(0).get());
    s.ta = op.hasAttribute("transpose_a") && op.getIntAttr("transpose_a") == 1;
    s.tb = op.hasAttribute("transpose_b") && op.getIntAttr("transpose_b") == 1;
    size_t ra = a.size();
    s.m = s.outShape[s.outShape.size() - 2];
    s.n = s.outShape.back();
    s.k = s.ta ? a[ra - 2] : a[ra - 1];
    if (a.size() == 3 && b.size() == 2 && !s.ta) {
        s.flattened = true;
        s.m *= a[0];
    } else {
        for (size_t d = 0; d + 2 < s.outShape.size(); ++d) s.batch *= s.outShape[d];
        s.aBatchStride = s.m * s.k;
        s.bBatchStride = b.size() == 2 ? 0 : s.k * s.n;
    }
    return s;
}

std::vector<LoweredKernel> lowerMatmul(const dlc::Kernel& k, const BufferAssignment& ba, const SlotTracker& names) {
    KernelBuilder kb(k.name, ba, names);
    kb.declare(k);
    const Operation* mm = nullptr;
    for (const Operation* op : k.ops) {
        if (op->getOpName() == "matmul") mm = op;
    }
    const MatmulShape s = matmulShape(*mm);
    const Value* A = mm->getOperand(0).get();
    const Value* B = mm->getOperand(1).get();
    const Value* C = mm->getResult(0).get();
    auto aAt = [&](const ExprPtr& b, const ExprPtr& i, const ExprPtr& kk) {
        ExprPtr idx = s.ta ? add(mul(kk, intConst(s.m)), i) : add(mul(i, intConst(s.k)), kk);
        return kb.loadValue(A, add(mul(b, intConst(s.aBatchStride)), idx));
    };
    auto bAt = [&](const ExprPtr& b, const ExprPtr& kk, const ExprPtr& j) {
        ExprPtr idx = s.tb ? add(mul(j, intConst(s.k)), kk) : add(mul(kk, intConst(s.n)), j);
        return kb.loadValue(B, add(mul(b, intConst(s.bBatchStride)), idx));
    };

    // The epilogue at output (b, i, j): the matmul's value is the accumulator; other members are elementwise. Values
    // of the output's shape use C's flat index; broadcast ones index by the output's multi-index.
    Shape batchShape(s.outShape.begin(), s.outShape.end() - 2);
    auto epilogue = [&](Body& body, const ExprPtr& b, const ExprPtr& i, const ExprPtr& j, ExprPtr accValue) {
        ExprPtr flat = add(mul(add(mul(b, intConst(s.m)), i), intConst(s.n)), j);
        std::vector<ExprPtr> idx;
        if (s.flattened) {
            int64_t rows = s.outShape[1];
            idx = {div(i, intConst(rows)), rem(i, intConst(rows)), j};
        } else {
            idx = unflatten(b, batchShape);
            idx.push_back(i);
            idx.push_back(j);
        }
        PointEmitter pe(kb, body, [&](const Value* v) {
            return shapeOf(v) == s.outShape ? flat : broadcastIndex(v, s.outShape, idx);
        });
        pe.bind(C, bindLocal(kb, body, "acc", std::move(accValue)));
        for (const Operation* op : k.ops) {
            if (op != mm) pe.member(*op);
        }
        pe.storeOutputs(k.outputs);
    };

    const bool outerProduct = !s.tb;
    const bool dotProduct = s.tb && !s.ta;
    // One register tile: mr x nr outputs starting at (iBase, jBase).
    auto tile = [&](const ExprPtr& b, const ExprPtr& iBase, int64_t mr, const ExprPtr& jBase, int64_t nr) {
        Body body;
        int acc = kb.local("acc", ScalarType::F32, {mr, nr}, MemScope::Register);
        std::string ii = kb.fresh("ii"), jj = kb.fresh("jj"), kk = kb.fresh("k");
        auto accIdx = [&] { return add(mul(var(ii), intConst(nr)), var(jj)); };
        body.push_back(forLoop(ii, mr, LoopKind::Unrolled,
                               {forLoop(jj, nr, LoopKind::Unrolled, {store(acc, accIdx(), floatConst(0.0))})}));
        ExprPtr i = add(iBase, var(ii));
        ExprPtr j = add(jBase, var(jj));
        Body update = {store(acc, accIdx(), fma(aAt(b, i, var(kk)), bAt(b, var(kk), j),
                                                  load(acc, accIdx(), ScalarType::F32)))};
        StmtPtr inner = forLoop(ii, mr, LoopKind::Unrolled, {forLoop(jj, nr, LoopKind::Unrolled, std::move(update))});
        // The outer-product form streams rows of B (contiguous in j); the dot-product form, for a transposed B,
        // vectorizes the reduction over k, where both A and B are contiguous.
        LoopKind kKind = dotProduct ? LoopKind::Vectorized : LoopKind::Serial;
        body.push_back(forLoop(kk, s.k, kKind, {inner}, dotProduct ? kVector : 1));
        Body epi;
        epilogue(epi, b, i, j, load(acc, accIdx(), ScalarType::F32));
        body.push_back(forLoop(ii, mr, LoopKind::Unrolled, {forLoop(jj, nr, LoopKind::Unrolled, std::move(epi))}));
        return body;
    };

    const int64_t mrTile = outerProduct || dotProduct ? kMR : 1;
    const int64_t nrTile = outerProduct ? kNR : (dotProduct ? 4 : 1);
    const int64_t mTiles = ceilDiv(s.m, mrTile);
    const int64_t mFull = s.m / mrTile;
    const int64_t nFull = s.n / nrTile;
    const int64_t mTail = s.m - mFull * mrTile;
    const int64_t nTail = s.n - nFull * nrTile;

    // The tiles of one row block of height mr.
    auto rowBlock = [&](const ExprPtr& b, const ExprPtr& iBase, int64_t mr) {
        Body body;
        if (nFull > 0) {
            std::string jt = kb.fresh("jt");
            body.push_back(forLoop(jt, nFull, LoopKind::Serial,
                                   tile(b, iBase, mr, mul(var(jt), intConst(nrTile)), nrTile)));
        }
        if (nTail > 0) {
            Body t = tile(b, iBase, mr, intConst(nFull * nrTile), nTail);
            body.insert(body.end(), t.begin(), t.end());
        }
        return body;
    };

    const bool parallel = 2 * s.batch * s.m * s.n * s.k >= kParallelWork * 16 && s.batch * mTiles > 1;
    Body perBlock;
    ExprPtr b = s.batch > 1 ? div(var("p"), intConst(mTiles)) : intConst(0);
    ExprPtr it = s.batch > 1 ? rem(var("p"), intConst(mTiles)) : var("p");
    ExprPtr bVar = bindLocal(kb, perBlock, "b", b);
    ExprPtr itVar = bindLocal(kb, perBlock, "it", it);
    ExprPtr iBase = mul(itVar, intConst(mrTile));
    if (mTail == 0) {
        Body full = rowBlock(bVar, iBase, mrTile);
        perBlock.insert(perBlock.end(), full.begin(), full.end());
    } else if (mFull == 0) {
        Body tail = rowBlock(bVar, iBase, mTail);
        perBlock.insert(perBlock.end(), tail.begin(), tail.end());
    } else {
        Body full = rowBlock(bVar, iBase, mrTile); // built in order, so variable names are deterministic
        Body tail = rowBlock(bVar, iBase, mTail);
        perBlock.push_back(ifThen(compare(CompareOp::LT, itVar, intConst(mFull)), std::move(full), std::move(tail)));
    }
    Body top = {forLoop("p", s.batch * mTiles, parallel ? LoopKind::Parallel : LoopKind::Serial, std::move(perBlock))};
    std::vector<LoweredKernel> out;
    out.push_back(kb.finish(std::move(top), k, true));
    return out;
}

// --- Reductions ----------------------------------------------------------------------------------------------------

struct Reduce {
    double init;
    std::function<ExprPtr(ExprPtr, ExprPtr)> combine;
};

Reduce reduceOf(const std::string& name) {
    if (name == "reduce_max") {
        return {-std::numeric_limits<double>::infinity(),
                [](ExprPtr a, ExprPtr b) { return binary(BinaryOp::MaxNaN, std::move(a), std::move(b)); }};
    }
    return {0.0, [](ExprPtr a, ExprPtr b) { return add(std::move(a), std::move(b)); }};
}

std::vector<LoweredKernel> lowerReduction(const dlc::Kernel& k, const BufferAssignment& ba, const SlotTracker& names) {
    KernelBuilder kb(k.name, ba, names);
    kb.declare(k);
    const Operation& op = *k.ops[0];
    const Value* x = op.getOperand(0).get();
    const Value* y = op.getResult(0).get();
    const Shape& xs = shapeOf(x);
    const int64_t rank = static_cast<int64_t>(xs.size());
    std::vector<bool> reduced(rank, false);
    for (int64_t a : op.getIntListAttr("axes")) reduced[a < 0 ? a + rank : a] = true;
    int64_t count = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (reduced[d]) count *= xs[d];
    }
    Reduce r = reduceOf(op.getOpName());
    const bool mean = op.getOpName() == "mean";
    auto finalize = [&](ExprPtr acc) {
        return mean ? div(std::move(acc), floatConst(static_cast<double>(count))) : acc;
    };

    // Reduced axes forming one contiguous block: x viewed as [outer, L, inner].
    int64_t first = -1, last = -1;
    for (int64_t d = 0; d < rank; ++d) {
        if (reduced[d]) {
            if (first < 0) first = d;
            last = d;
        }
    }
    bool contiguous = first >= 0;
    for (int64_t d = first; contiguous && d <= last; ++d) contiguous = reduced[d];
    Body top;
    if (first < 0) { // no axes: a copy
        top.push_back(forLoop("e", numel(xs), LoopKind::Vectorized,
                              {store(kb.param(y), var("e"), kb.loadValue(x, var("e")))}, kVector));
    } else if (contiguous) {
        int64_t outer = 1, L = 1, inner = 1;
        for (int64_t d = 0; d < first; ++d) outer *= xs[d];
        for (int64_t d = first; d <= last; ++d) L *= xs[d];
        for (int64_t d = last + 1; d < rank; ++d) inner *= xs[d];
        const bool parallel = numel(xs) >= kParallelWork;
        if (inner == 1) {
            // Rows: a register accumulator over the contiguous reduced block.
            Body row = {let("acc", ScalarType::F32, floatConst(r.init)),
                        forLoop("l", L, LoopKind::Vectorized,
                                {assign("acc", r.combine(var("acc", ScalarType::F32),
                                                          kb.loadValue(x, add(mul(var("o"), intConst(L)), var("l")))))},
                                kVector),
                        store(kb.param(y), var("o"), finalize(var("acc", ScalarType::F32)))};
            top.push_back(forLoop("o", outer, parallel && outer > 1 ? LoopKind::Parallel : LoopKind::Serial, std::move(row)));
        } else {
            // Columns: each output row is accumulated in place, streaming contiguous rows of x.
            ExprPtr yIdx = add(mul(var("o"), intConst(inner)), var("q"));
            Body body = {forLoop("q", inner, LoopKind::Vectorized, {store(kb.param(y), yIdx, floatConst(r.init))}, kVector),
                         forLoop("l", L, LoopKind::Serial,
                                 {forLoop("q", inner, LoopKind::Vectorized,
                                          {store(kb.param(y), yIdx,
                                                 r.combine(kb.loadValue(y, yIdx),
                                                           kb.loadValue(x, add(mul(add(mul(var("o"), intConst(L)), var("l")),
                                                                                   intConst(inner)),
                                                                               var("q")))))},
                                          kVector)})};
            if (mean) {
                body.push_back(forLoop("q", inner, LoopKind::Vectorized,
                                       {store(kb.param(y), yIdx, finalize(kb.loadValue(y, yIdx)))}, kVector));
            }
            top.push_back(forLoop("o", outer, parallel && outer > 1 ? LoopKind::Parallel : LoopKind::Serial, std::move(body)));
        }
    } else {
        // Any other set of axes: initialize, accumulate over every element of x, finalize.
        Shape kept;
        for (int64_t d = 0; d < rank; ++d) {
            if (!reduced[d]) kept.push_back(xs[d]);
        }
        auto keptStrides = stridesOf(kept);
        std::vector<ExprPtr> idx;
        ExprPtr xIdx = intConst(0), yIdx = intConst(0);
        auto xst = stridesOf(xs);
        for (int64_t d = 0, kd = 0; d < rank; ++d) {
            ExprPtr v = var("x" + std::to_string(d));
            xIdx = add(xIdx, mul(v, intConst(xst[d])));
            if (!reduced[d]) yIdx = add(yIdx, mul(v, intConst(keptStrides[kd++])));
        }
        int64_t outCount = numel(kept);
        top.push_back(forLoop("e", outCount, LoopKind::Vectorized, {store(kb.param(y), var("e"), floatConst(r.init))}, kVector));
        Body acc = {store(kb.param(y), yIdx, r.combine(kb.loadValue(y, yIdx), kb.loadValue(x, xIdx)))};
        StmtPtr loop = forLoop("x" + std::to_string(rank - 1), xs[rank - 1], LoopKind::Serial, std::move(acc));
        for (int64_t d = rank - 2; d >= 0; --d) loop = forLoop("x" + std::to_string(d), xs[d], LoopKind::Serial, {loop});
        top.push_back(loop);
        if (mean) {
            top.push_back(forLoop("e", outCount, LoopKind::Vectorized,
                                  {store(kb.param(y), var("e"), finalize(kb.loadValue(y, var("e"))))}, kVector));
        }
    }
    std::vector<LoweredKernel> out;
    out.push_back(kb.finish(std::move(top), k, true));
    return out;
}

// --- Row kernels ---------------------------------------------------------------------------------------------------

std::vector<LoweredKernel> lowerRow(const dlc::Kernel& k, const BufferAssignment& ba, const SlotTracker& names) {
    KernelBuilder kb(k.name, ba, names);
    kb.declare(k);
    const Operation& op = *k.ops[0];
    const std::string& name = op.getOpName();
    const Value* x = op.getOperand(0).get();
    const Value* y = op.getResult(0).get();
    const Shape& xs = shapeOf(x);
    const int64_t rank = static_cast<int64_t>(xs.size());
    const ExprPtr negInf = floatConst(-std::numeric_limits<double>::infinity());
    auto f32 = [](const std::string& n) { return var(n, ScalarType::F32); };
    Body top;

    if (name == "softmax" || name == "log_softmax") {
        int64_t axis = op.getIntAttr("axis");
        if (axis < 0) axis += rank;
        int64_t outer = 1, L = xs[axis], inner = 1;
        for (int64_t d = 0; d < axis; ++d) outer *= xs[d];
        for (int64_t d = axis + 1; d < rank; ++d) inner *= xs[d];
        auto at = [&](const ExprPtr& l) { return add(mul(add(mul(var("o"), intConst(L)), l), intConst(inner)), var("q")); };
        LoopKind lk = inner == 1 ? LoopKind::Vectorized : LoopKind::Serial;
        Body row = {let("m", ScalarType::F32, negInf),
                    forLoop("l", L, lk, {assign("m", binary(BinaryOp::MaxNaN, f32("m"), kb.loadValue(x, at(var("l")))))}, kVector),
                    let("s", ScalarType::F32, floatConst(0.0)),
                    forLoop("l", L, lk, {assign("s", add(f32("s"), unary(UnaryOp::Exp, sub(kb.loadValue(x, at(var("l"))), f32("m")))))}, kVector)};
        if (name == "softmax") {
            row.push_back(forLoop("l", L, lk, {store(kb.param(y), at(var("l")),
                                                      div(unary(UnaryOp::Exp, sub(kb.loadValue(x, at(var("l"))), f32("m"))), f32("s")))}, kVector));
        } else {
            row.push_back(let("lse", ScalarType::F32, unary(UnaryOp::Log, f32("s"))));
            row.push_back(forLoop("l", L, lk, {store(kb.param(y), at(var("l")),
                                                      sub(sub(kb.loadValue(x, at(var("l"))), f32("m")), f32("lse")))}, kVector));
        }
        Body perOuter = {forLoop("q", inner, LoopKind::Serial, std::move(row))};
        LoopKind ok = numel(xs) >= kParallelWork && outer > 1 ? LoopKind::Parallel : LoopKind::Serial;
        top.push_back(forLoop("o", outer, ok, std::move(perOuter)));
    } else if (name == "layernorm") {
        const Value* gamma = op.getOperand(1).get();
        const Value* beta = op.getOperand(2).get();
        int64_t D = xs.back();
        int64_t rows = numel(xs) / std::max<int64_t>(D, 1);
        auto at = [&](const ExprPtr& d) { return add(mul(var("r"), intConst(D)), d); };
        Body row = {let("s", ScalarType::F32, floatConst(0.0)),
                    forLoop("d", D, LoopKind::Vectorized, {assign("s", add(f32("s"), kb.loadValue(x, at(var("d")))))}, kVector),
                    let("mu", ScalarType::F32, div(f32("s"), floatConst(static_cast<double>(D)))),
                    let("v", ScalarType::F32, floatConst(0.0)),
                    forLoop("d", D, LoopKind::Vectorized,
                            {let("c", ScalarType::F32, sub(kb.loadValue(x, at(var("d"))), f32("mu"))),
                             assign("v", add(f32("v"), mul(f32("c"), f32("c"))))}, kVector),
                    let("rstd", ScalarType::F32, unary(UnaryOp::Rsqrt, add(div(f32("v"), floatConst(static_cast<double>(D))),
                                                                          floatConst(op.getDoubleAttr("eps"))))),
                    forLoop("d", D, LoopKind::Vectorized,
                            {store(kb.param(y), at(var("d")),
                                   add(mul(mul(sub(kb.loadValue(x, at(var("d"))), f32("mu")), f32("rstd")),
                                           kb.loadValue(gamma, var("d"))),
                                       kb.loadValue(beta, var("d"))))}, kVector)};
        LoopKind rk = numel(xs) >= kParallelWork && rows > 1 ? LoopKind::Parallel : LoopKind::Serial;
        top.push_back(forLoop("r", rows, rk, std::move(row)));
    } else if (name == "cross_entropy") {
        const Value* t = op.getOperand(1).get();
        int64_t C = xs.back();
        int64_t rows = numel(xs) / std::max<int64_t>(C, 1);
        auto at = [&](const ExprPtr& c) { return add(mul(var("r"), intConst(C)), c); };
        Body row = {let("m", ScalarType::F32, negInf),
                    forLoop("c", C, LoopKind::Vectorized, {assign("m", binary(BinaryOp::MaxNaN, f32("m"), kb.loadValue(x, at(var("c")))))}, kVector),
                    let("s", ScalarType::F32, floatConst(0.0)),
                    forLoop("c", C, LoopKind::Vectorized, {assign("s", add(f32("s"), unary(UnaryOp::Exp, sub(kb.loadValue(x, at(var("c"))), f32("m")))))}, kVector),
                    let("target", ScalarType::I64, cast(ScalarType::I64, kb.loadValue(t, var("r")))),
                    // A label outside [0, C) makes the loss NaN rather than reading out of bounds.
                    guardIndex("target", C,
                               {assign("total", add(f32("total"), sub(add(unary(UnaryOp::Log, f32("s")), f32("m")),
                                                                     kb.loadValue(x, at(var("target"))))))},
                               {assign("total", floatConst(std::numeric_limits<double>::quiet_NaN()))})};
        top.push_back(let("total", ScalarType::F32, floatConst(0.0)));
        top.push_back(forLoop("r", rows, LoopKind::Serial, std::move(row)));
        top.push_back(store(kb.param(y), intConst(0),
                            rows == 0 ? floatConst(std::numeric_limits<double>::quiet_NaN())
                                      : div(f32("total"), floatConst(static_cast<double>(rows)))));
    } else {
        throw IRError("lowering: no row kernel for '" + name + "'");
    }
    std::vector<LoweredKernel> out;
    out.push_back(kb.finish(std::move(top), k, true));
    return out;
}

// --- Gather, scatter, transpose, fill ------------------------------------------------------------------------------


std::vector<LoweredKernel> lowerOther(const dlc::Kernel& k, const BufferAssignment& ba, const SlotTracker& names) {
    KernelBuilder kb(k.name, ba, names);
    kb.declare(k);
    const Operation& op = *k.ops[0];
    const std::string& name = op.getOpName();
    const Value* y = op.getResult(0).get();
    Body top;
    if (name == "embedding") {
        const Value* table = op.getOperand(0).get();
        const Value* ids = op.getOperand(1).get();
        int64_t D = shapeOf(table)[1];
        int64_t P = numel(shapeOf(ids));
        int64_t V = shapeOf(table)[0];
        ExprPtr outIdx = add(mul(var("p"), intConst(D)), var("d"));
        Body copy = {forLoop("d", D, LoopKind::Vectorized,
                             {store(kb.param(y), outIdx, kb.loadValue(table, add(mul(var("id"), intConst(D)), var("d"))))},
                             kVector)};
        Body zero = {forLoop("d", D, LoopKind::Vectorized, {store(kb.param(y), outIdx, floatConst(0.0))}, kVector)};
        Body row = {let("id", ScalarType::I64, cast(ScalarType::I64, kb.loadValue(ids, var("p")))),
                    guardIndex("id", V, std::move(copy), std::move(zero))};
        top.push_back(forLoop("p", P, P * D >= kParallelWork ? LoopKind::Parallel : LoopKind::Serial, std::move(row)));
    } else if (name == "onehot") {
        const Value* ids = op.getOperand(0).get();
        int64_t C = op.getIntAttr("num_classes");
        int64_t P = numel(shapeOf(ids));
        Body row = {let("id", ScalarType::I64, cast(ScalarType::I64, kb.loadValue(ids, var("p")))),
                    forLoop("c", C, LoopKind::Vectorized,
                            {store(kb.param(y), add(mul(var("p"), intConst(C)), var("c")),
                                   select(compare(CompareOp::EQ, var("c"), var("id")), floatConst(1.0), floatConst(0.0)))}, kVector)};
        top.push_back(forLoop("p", P, LoopKind::Serial, std::move(row)));
    } else if (name == "scatter_add") {
        const Value* ids = op.getOperand(0).get();
        const Value* upd = op.getOperand(1).get();
        const Shape& ys = shapeOf(y);
        int64_t D = ys[1];
        int64_t P = numel(shapeOf(ids));
        top.push_back(forLoop("e", numel(ys), LoopKind::Vectorized, {store(kb.param(y), var("e"), floatConst(0.0))}, kVector));
        ExprPtr yIdx = add(mul(var("id"), intConst(D)), var("d"));
        Body accumulate = {forLoop("d", D, LoopKind::Vectorized,
                                   {store(kb.param(y), yIdx, add(kb.loadValue(y, yIdx),
                                                                 kb.loadValue(upd, add(mul(var("p"), intConst(D)), var("d")))))},
                                   kVector)};
        Body row = {let("id", ScalarType::I64, cast(ScalarType::I64, kb.loadValue(ids, var("p")))),
                    guardIndex("id", ys[0], std::move(accumulate))};
        top.push_back(forLoop("p", P, LoopKind::Serial, std::move(row)));
    } else if (name == "transpose") {
        const Value* x = op.getOperand(0).get();
        const auto& perm = op.getIntListAttr("perm");
        const Shape& ys = shapeOf(y);
        auto xst = stridesOf(shapeOf(x));
        auto yst = stridesOf(ys);
        int64_t rank = static_cast<int64_t>(ys.size());
        if (rank == 0) {
            top.push_back(store(kb.param(y), intConst(0), kb.loadValue(x, intConst(0))));
        } else {
            ExprPtr xIdx = intConst(0), yIdx = intConst(0);
            for (int64_t d = 0; d < rank; ++d) {
                ExprPtr v = var("t" + std::to_string(d));
                yIdx = add(yIdx, mul(v, intConst(yst[d])));
                xIdx = add(xIdx, mul(v, intConst(xst[perm[d]])));
            }
            StmtPtr loop = forLoop("t" + std::to_string(rank - 1), ys[rank - 1], LoopKind::Vectorized,
                                   {store(kb.param(y), yIdx, kb.loadValue(x, xIdx))}, kVector);
            for (int64_t d = rank - 2; d >= 0; --d) {
                LoopKind kind = d == 0 && numel(ys) >= kParallelWork ? LoopKind::Parallel : LoopKind::Serial;
                loop = forLoop("t" + std::to_string(d), ys[d], kind, {loop});
            }
            top.push_back(loop);
        }
    } else if (name == "constant") {
        ExprPtr value = typeOf(y) == ScalarType::F32 ? floatConst(op.getDoubleAttr("value"))
                                                     : cast(typeOf(y), intConst(static_cast<int64_t>(op.getDoubleAttr("value"))));
        top.push_back(forLoop("e", numel(shapeOf(y)), LoopKind::Vectorized, {store(kb.param(y), var("e"), value)}, kVector));
    } else {
        throw IRError("lowering: no kernel for '" + name + "'");
    }
    std::vector<LoweredKernel> out;
    out.push_back(kb.finish(std::move(top), k, true));
    return out;
}

} // namespace

std::vector<LoweredKernel> lowerFunction(const Function& func, const std::vector<dlc::Kernel>& kernels,
                                         const BufferAssignment& buffers, Target target) {
    if (target != Target::CPU) throw IRError("lowering: the GPU schedules arrive in P5");
    SlotTracker names(func);
    std::vector<LoweredKernel> out;
    for (const dlc::Kernel& k : kernels) {
        std::vector<LoweredKernel> lowered;
        switch (k.kind) {
            case KernelKind::Elementwise: lowered = lowerElementwise(k, buffers, names); break;
            case KernelKind::Matmul: lowered = lowerMatmul(k, buffers, names); break;
            case KernelKind::Reduction: lowered = lowerReduction(k, buffers, names); break;
            case KernelKind::Row: lowered = lowerRow(k, buffers, names); break;
            default: lowered = lowerOther(k, buffers, names); break;
        }
        for (auto& l : lowered) {
            // Loop passes: guards for data-dependent indices, bounds-check elimination, then the dependence check.
            l.bounds = checkBounds(l.kernel);
            verifySchedule(l.kernel);
            out.push_back(std::move(l));
        }
    }
    return out;
}

} // namespace dlc::loopir
