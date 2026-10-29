#include "dlc/passes/Fusion.h"
#include "dlc/ir/Error.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_set>

namespace dlc {

std::string kernelKindName(KernelKind kind) {
    switch (kind) {
        case KernelKind::Matmul: return "matmul";
        case KernelKind::Elementwise: return "elementwise";
        case KernelKind::Reduction: return "reduction";
        case KernelKind::Row: return "row";
        case KernelKind::Gather: return "gather";
        case KernelKind::Scatter: return "scatter";
        case KernelKind::Transpose: return "transpose";
        case KernelKind::Fill: return "fill";
    }
    return "unknown";
}

bool isElementwiseOp(const std::string& name) {
    static const std::set<std::string> ops = {"add", "sub", "mul", "div", "relu", "gelu", "exp", "log", "tanh",
                                              "sqrt", "rsqrt", "cmp_gt", "select", "cast", "broadcast_to"};
    return ops.count(name) != 0;
}

namespace {

KernelKind kindOf(const std::string& name) {
    if (name == "matmul") return KernelKind::Matmul;
    if (isElementwiseOp(name)) return KernelKind::Elementwise;
    if (name == "sum" || name == "mean" || name == "reduce_max") return KernelKind::Reduction;
    if (name == "softmax" || name == "log_softmax" || name == "layernorm" || name == "cross_entropy") {
        return KernelKind::Row;
    }
    if (name == "embedding" || name == "onehot") return KernelKind::Gather;
    if (name == "scatter_add") return KernelKind::Scatter;
    if (name == "transpose") return KernelKind::Transpose;
    if (name == "constant") return KernelKind::Fill;
    throw IRError("fusion: no kernel class for '" + name + "'");
}

const Shape& shapeOf(const Value* value) { return static_cast<const TensorType&>(*value->getType()).getShape(); }

bool isSplat(const Value* value) {
    const Operation* def = value->getDefiningOp();
    return def && def->getOpName() == "constant";
}

// The ops that read `value`, looking through reshapes (which alias their operand).
void collectUsers(const Value* value, std::vector<Operation*>& users) {
    for (Operation* user : value->getUses()) {
        if (user->getOpName() == "reshape") collectUsers(user->getResult(0).get(), users);
        else users.push_back(user);
    }
}

class Grouper {
public:
    Grouper(const Function& func, bool fuse) : func_(func), fuse_(fuse) {}

    void run(std::vector<Kernel>& kernels) {
        const auto& ops = func_.getEntryBlock()->getOperations();
        for (size_t i = 0; i < ops.size(); ++i) position_[ops[i].get()] = i;

        // One group per op that needs a kernel. Splat constants are inlined into elementwise code, so they get a
        // (Fill) kernel only when something reads them from memory.
        for (const auto& op : ops) {
            const std::string& name = op->getOpName();
            if (name == "return" || name == "reshape") continue;
            if (name == "constant" && !readFromMemory(op->getResult(0).get())) continue;
            newGroup(op.get());
        }
        // Grow elementwise groups and matmul epilogues along producer → consumer edges, in program order. Matmul
        // groups go first, so an elementwise op that consumes a matmul result joins its epilogue.
        for (const auto& op : ops) {
            if (!fuse_ || !isElementwiseOp(op->getOpName())) continue;
            const Shape& shape = shapeOf(op->getResult(0).get());
            std::vector<size_t> producers;
            for (const auto& operand : op->getOperands()) {
                Operation* producer = operand->getDefiningOp();
                if (producer && groupOf_.count(producer) && shapeOf(operand.get()) == shape) {
                    producers.push_back(find(groupOf_.at(producer)));
                }
            }
            std::stable_partition(producers.begin(), producers.end(),
                                  [&](size_t g) { return groups_[g].kind == KernelKind::Matmul; });
            for (size_t producer : producers) {
                size_t target = find(producer);
                size_t self = find(groupOf_.at(op.get()));
                if (target == self || !compatible(target, self, shape) || createsCycle(target, self)) continue;
                merge(target, self);
            }
        }
        build(kernels);
    }

private:
    struct Group {
        KernelKind kind;
        Shape shape;
        std::vector<Operation*> ops;
        size_t parent;
        Operation* matmul{nullptr};
    };

    bool readFromMemory(const Value* constant) const {
        for (Operation* user : constant->getUses()) {
            if (!isElementwiseOp(user->getOpName())) return true; // matmul operands, reductions, reshape, return...
        }
        return false;
    }

    void newGroup(Operation* op) {
        size_t id = groups_.size();
        KernelKind kind = kindOf(op->getOpName());
        Shape shape = op->getNumResults() ? shapeOf(op->getResult(0).get()) : Shape{};
        groups_.push_back({kind, shape, {op}, id, kind == KernelKind::Matmul ? op : nullptr});
        groupOf_[op] = id;
    }

    size_t find(size_t g) {
        while (groups_[g].parent != g) g = groups_[g].parent = groups_[groups_[g].parent].parent;
        return g;
    }

    // Ops reachable from `op` along def-use edges.
    const std::unordered_set<const Operation*>& descendants(const Operation* op) {
        auto [it, inserted] = descendants_.try_emplace(op);
        if (!inserted) return it->second;
        std::vector<const Operation*> stack{op};
        while (!stack.empty()) {
            const Operation* cur = stack.back();
            stack.pop_back();
            for (const auto& result : cur->getResults()) {
                for (Operation* user : result->getUses()) {
                    if (it->second.insert(user).second) stack.push_back(user);
                }
            }
        }
        return it->second;
    }

    // Elementwise groups of one shape merge; a matmul group absorbs elementwise ops as its epilogue, evaluated per
    // output element after the accumulation, so none of them may feed the matmul.
    bool compatible(size_t a, size_t b, const Shape& shape) {
        const Group& ga = groups_[a];
        const Group& gb = groups_[b];
        for (const Group* g : {&ga, &gb}) {
            if (g->kind != KernelKind::Elementwise && g->kind != KernelKind::Matmul) return false;
            if (g->shape != shape) return false;
        }
        if (ga.kind == KernelKind::Matmul && gb.kind == KernelKind::Matmul) return false;
        if (ga.kind == KernelKind::Elementwise && gb.kind == KernelKind::Elementwise) return true;
        const Group& mm = ga.kind == KernelKind::Matmul ? ga : gb;
        const Group& ew = ga.kind == KernelKind::Matmul ? gb : ga;
        return std::none_of(ew.ops.begin(), ew.ops.end(),
                            [&](const Operation* op) { return descendants(op).count(mm.matmul) != 0; });
    }

    // The groups whose members read a result of group `g` (through reshapes).
    std::vector<size_t> successors(size_t g) {
        std::vector<size_t> out;
        for (Operation* op : groups_[g].ops) {
            for (const auto& result : op->getResults()) {
                std::vector<Operation*> users;
                collectUsers(result.get(), users);
                for (Operation* user : users) {
                    auto it = groupOf_.find(user);
                    if (it == groupOf_.end()) continue;
                    size_t s = find(it->second);
                    if (s != g) out.push_back(s);
                }
            }
        }
        return out;
    }

    // True if `to` is reachable from `from` through some third group: merging them would make the kernel graph cyclic.
    bool reachesThroughOther(size_t from, size_t to) {
        std::vector<size_t> stack;
        std::unordered_set<size_t> seen;
        for (size_t s : successors(from)) {
            if (s != to && seen.insert(s).second) stack.push_back(s);
        }
        while (!stack.empty()) {
            size_t g = stack.back();
            stack.pop_back();
            for (size_t s : successors(g)) {
                if (s == to) return true;
                if (seen.insert(s).second) stack.push_back(s);
            }
        }
        return false;
    }

    bool createsCycle(size_t a, size_t b) { return reachesThroughOther(a, b) || reachesThroughOther(b, a); }

    void merge(size_t into, size_t from) {
        Group& dst = groups_[into];
        Group& src = groups_[from];
        if (src.kind == KernelKind::Matmul) {
            dst.kind = KernelKind::Matmul;
            dst.matmul = src.matmul;
        }
        dst.ops.insert(dst.ops.end(), src.ops.begin(), src.ops.end());
        src.ops.clear();
        src.parent = into;
    }

    void build(std::vector<Kernel>& kernels) {
        std::vector<size_t> roots;
        for (size_t g = 0; g < groups_.size(); ++g) {
            if (find(g) == g) roots.push_back(g);
        }
        for (size_t g : roots) {
            auto& ops = groups_[g].ops;
            std::sort(ops.begin(), ops.end(), [&](Operation* a, Operation* b) { return position_[a] < position_[b]; });
        }
        // Topological order of the kernel graph (Kahn's algorithm), the kernel with the earliest member first.
        std::unordered_map<size_t, size_t> indegree;
        std::unordered_map<size_t, std::set<size_t>> edges;
        for (size_t g : roots) indegree[g] = 0;
        for (size_t g : roots) {
            for (size_t s : successors(g)) {
                if (edges[g].insert(s).second) ++indegree[s];
            }
        }
        auto later = [&](size_t a, size_t b) {
            return position_[groups_[a].ops.front()] > position_[groups_[b].ops.front()];
        };
        std::priority_queue<size_t, std::vector<size_t>, decltype(later)> ready(later);
        for (size_t g : roots) {
            if (indegree[g] == 0) ready.push(g);
        }
        std::vector<size_t> order;
        while (!ready.empty()) {
            size_t g = ready.top();
            ready.pop();
            order.push_back(g);
            for (size_t s : edges[g]) {
                if (--indegree[s] == 0) ready.push(s);
            }
        }
        if (order.size() != roots.size()) throw IRError("fusion: the kernel graph has a cycle");

        for (size_t g : order) {
            const Group& group = groups_[g];
            Kernel k;
            k.id = static_cast<int64_t>(kernels.size());
            k.kind = group.kind;
            k.ops = group.ops;
            k.iterationShape = group.shape;
            std::unordered_set<const Operation*> members(group.ops.begin(), group.ops.end());
            std::unordered_set<const Value*> seenInputs;
            for (Operation* op : group.ops) {
                bool inlines = isElementwiseOp(op->getOpName());
                for (const auto& operand : op->getOperands()) {
                    const Operation* def = operand->getDefiningOp();
                    if (def && members.count(def)) continue;
                    if (inlines && isSplat(operand.get())) continue;
                    if (seenInputs.insert(operand.get()).second) k.inputs.push_back(operand.get());
                }
                for (const auto& result : op->getResults()) {
                    const auto& uses = result->getUses();
                    if (std::any_of(uses.begin(), uses.end(), [&](Operation* u) { return !members.count(u); })) {
                        k.outputs.push_back(result.get());
                    }
                }
            }
            k.name = "k" + std::to_string(k.id);
            for (size_t i = 0; i < group.ops.size() && i < 4; ++i) k.name += "_" + group.ops[i]->getOpName();
            if (group.ops.size() > 4) k.name += "_x" + std::to_string(group.ops.size());
            kernels.push_back(std::move(k));
        }
    }

    const Function& func_;
    bool fuse_{true};
    std::vector<Group> groups_;
    std::unordered_map<const Operation*, size_t> groupOf_;
    std::unordered_map<const Operation*, size_t> position_;
    std::unordered_map<const Operation*, std::unordered_set<const Operation*>> descendants_;
};

} // namespace

bool FusionPass::runOnFunction(Function& func) {
    kernels_.clear();
    kernelOfValue_.clear();
    if (func.getEntryBlock()->getOperations().empty()) return false;
    Grouper(func, fuse_).run(kernels_);
    for (size_t k = 0; k < kernels_.size(); ++k) {
        for (Operation* op : kernels_[k].ops) {
            for (const auto& result : op->getResults()) kernelOfValue_[result.get()] = k;
            if (annotate_) {
                op->setAttribute("fusion.group", kernels_[k].id);
                op->setAttribute("fusion.kernel", kernels_[k].name);
            }
        }
    }
    return annotate_ && !kernels_.empty();
}

const Kernel* FusionPass::kernelOf(const Value* value) const {
    auto it = kernelOfValue_.find(value);
    return it == kernelOfValue_.end() ? nullptr : &kernels_[it->second];
}

} // namespace dlc
