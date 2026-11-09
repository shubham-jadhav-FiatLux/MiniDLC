#include "dlc/loopir/Transforms.h"
#include "dlc/ir/Error.h"
#include "dlc/loopir/Analysis.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace dlc::loopir {

ExprPtr rebuild(const Expr& e, std::vector<ExprPtr> args) {
    switch (e.kind) {
        case ExprKind::Int:
        case ExprKind::Float:
        case ExprKind::Var: return std::make_shared<Expr>(e);
        case ExprKind::Load: return load(e.buffer, std::move(args[0]), e.type);
        case ExprKind::Binary:
            if (e.binary == BinaryOp::Fma) return fma(std::move(args[0]), std::move(args[1]), std::move(args[2]));
            return binary(e.binary, std::move(args[0]), std::move(args[1]));
        case ExprKind::Unary: return unary(e.unary, std::move(args[0]));
        case ExprKind::Select: return select(std::move(args[0]), std::move(args[1]), std::move(args[2]));
        case ExprKind::Compare: return compare(e.compare, std::move(args[0]), std::move(args[1]));
        case ExprKind::Cast: return cast(e.type, std::move(args[0]));
        case ExprKind::Shuffle:
            return e.butterfly ? shuffleXor(std::move(args[0]), std::move(args[1])) : shuffleDown(std::move(args[0]), std::move(args[1]));
    }
    throw IRError("loop IR: unknown expression");
}

namespace {

// Post-order rewrite: the operands first (a changed operand rebuilds its parent through the constructors, which
// fold), then `f` on the node.
template <class F> ExprPtr transform(const ExprPtr& e, F& f) {
    if (!e) return e;
    ExprPtr node = e;
    if (!e->args.empty()) {
        std::vector<ExprPtr> args;
        bool changed = false;
        for (const auto& a : e->args) {
            args.push_back(transform(a, f));
            changed = changed || args.back() != a;
        }
        if (changed) node = rebuild(*e, std::move(args));
    }
    return f(node);
}

template <class F> void transformBody(Body& stmts, F& f) {
    for (auto& s : stmts) {
        if (s->value) s->value = transform(s->value, f);
        if (s->index) s->index = transform(s->index, f);
        if (s->cond) s->cond = transform(s->cond, f);
        transformBody(s->body, f);
        transformBody(s->elseBody, f);
    }
}

// Safe to evaluate anywhere, any number of times: no memory, no other lanes, and no integer division that could trap
// (only by a nonzero constant) — a hoisted value runs even when its loop runs zero times.
bool pure(const ExprPtr& e) {
    if (e->kind == ExprKind::Load || e->kind == ExprKind::Shuffle) return false;
    if (e->kind == ExprKind::Binary && e->type != ScalarType::F32 && (e->binary == BinaryOp::Div || e->binary == BinaryOp::Rem) &&
        (e->args[1]->kind != ExprKind::Int || e->args[1]->intValue == 0)) {
        return false;
    }
    return std::all_of(e->args.begin(), e->args.end(), pure);
}

void collectVars(const ExprPtr& e, std::set<std::string>& out) {
    if (e->kind == ExprKind::Var) out.insert(e->name);
    for (const auto& a : e->args) collectVars(a, out);
}

// Names bound (Let, loop variables) and assigned anywhere in `stmts`.
struct Bindings {
    std::map<std::string, int> bound;
    std::set<std::string> assigned;
};

void collectBindings(const Body& stmts, Bindings& out) {
    for (const auto& s : stmts) {
        if (s->kind == StmtKind::For) ++out.bound[s->var];
        if (s->kind == StmtKind::Let) ++out.bound[s->name];
        if (s->kind == StmtKind::Assign) out.assigned.insert(s->name);
        collectBindings(s->body, out);
        collectBindings(s->elseBody, out);
    }
}

// --- constant propagation ---

int propagateConstants(Kernel& k) {
    int total = 0;
    for (bool changed = true; changed;) {
        changed = false;
        Bindings b;
        collectBindings(k.body, b);
        std::function<const Stmt*(const Body&)> find = [&](const Body& stmts) -> const Stmt* {
            for (const auto& s : stmts) {
                if (s->kind == StmtKind::Let && s->value->kind == ExprKind::Int && b.bound[s->name] == 1 &&
                    !b.assigned.count(s->name)) {
                    return s.get();
                }
                if (const Stmt* inner = find(s->body)) return inner;
                if (const Stmt* inner = find(s->elseBody)) return inner;
            }
            return nullptr;
        };
        if (const Stmt* s = find(k.body)) {
            std::string name = s->name;
            ExprPtr value = s->value;
            removeLets(k.body, name);
            substitute(k.body, name, value);
            ++total;
            changed = true;
        }
    }
    return total;
}

// --- division and remainder folding with range facts ---

class DivRemFolder {
public:
    DivRemFolder(const std::set<std::string>& assigned) : assigned_(assigned) {}

    int run(Body& stmts) {
        body(stmts);
        return folded_;
    }

private:
    ExprPtr fold(const ExprPtr& e) {
        auto f = [this](const ExprPtr& x) -> ExprPtr {
            if (x->kind != ExprKind::Binary || (x->binary != BinaryOp::Div && x->binary != BinaryOp::Rem)) return x;
            if (x->type != ScalarType::I64 || x->args[1]->kind != ExprKind::Int || x->args[1]->intValue <= 0) return x;
            const int64_t c = x->args[1]->intValue;
            Interval r = rangeOf(x->args[0], env_);
            if (!r.known() || r.lo < 0 || r.lo / c != r.hi / c) return x;
            const int64_t q = r.lo / c;
            ++folded_;
            if (x->binary == BinaryOp::Div) return intConst(q);
            return q == 0 ? x->args[0] : sub(x->args[0], intConst(q * c));
        };
        return transform(e, f);
    }

    void body(Body& stmts) {
        env_.push();
        for (auto& s : stmts) {
            if (s->value) s->value = fold(s->value);
            if (s->index) s->index = fold(s->index);
            if (s->cond) s->cond = fold(s->cond);
            switch (s->kind) {
                case StmtKind::For:
                    if (s->extent <= 0) break;
                    env_.push();
                    env_.set(s->var, {0, s->extent - 1});
                    body(s->body);
                    env_.pop();
                    break;
                case StmtKind::Let:
                    if (s->type != ScalarType::F32) {
                        env_.set(s->name, assigned_.count(s->name) ? Interval::unknown() : rangeOf(s->value, env_));
                    }
                    break;
                case StmtKind::If:
                    for (bool taken : {true, false}) {
                        env_.push();
                        if (refineRange(env_, s->cond, taken)) body(taken ? s->body : s->elseBody);
                        env_.pop();
                    }
                    break;
                default: break;
            }
        }
        env_.pop();
    }

    const std::set<std::string>& assigned_;
    RangeEnv env_;
    int folded_{0};
};

// --- loop-invariant code motion ---

class Hoister {
public:
    explicit Hoister(const Bindings& b) : b_(b) {}

    int run(Body& stmts) {
        walk(stmts);
        return hoisted_;
    }

private:
    bool movable(const Stmt& let, const std::set<std::string>& definedInLoop) const {
        if (let.kind != StmtKind::Let || !pure(let.value)) return false;
        auto bound = b_.bound.find(let.name);
        if (bound == b_.bound.end() || bound->second != 1 || b_.assigned.count(let.name)) return false;
        std::set<std::string> uses;
        collectVars(let.value, uses);
        return std::none_of(uses.begin(), uses.end(), [&](const std::string& u) { return definedInLoop.count(u) != 0; });
    }

    void walk(Body& stmts) {
        for (size_t i = 0; i < stmts.size(); ++i) {
            StmtPtr s = stmts[i];
            walk(s->body); // inner loops first: what they hoist may move further out
            walk(s->elseBody);
            if (s->kind != StmtKind::For || s->loop == LoopKind::Parallel || s->loop == LoopKind::Bound || s->extent <= 0) continue;
            Body moved;
            for (bool changed = true; changed;) {
                changed = false;
                Bindings inLoop;
                collectBindings(s->body, inLoop);
                std::set<std::string> defined(inLoop.assigned.begin(), inLoop.assigned.end());
                for (const auto& [name, count] : inLoop.bound) defined.insert(name);
                defined.insert(s->var);
                for (size_t j = 0; j < s->body.size(); ++j) {
                    // The Let itself is bound in the loop; it may move if nothing else it uses is.
                    std::set<std::string> others = defined;
                    if (s->body[j]->kind == StmtKind::Let) others.erase(s->body[j]->name);
                    if (movable(*s->body[j], others)) {
                        moved.push_back(s->body[j]);
                        s->body.erase(s->body.begin() + static_cast<std::ptrdiff_t>(j));
                        ++hoisted_;
                        changed = true;
                        break;
                    }
                }
            }
            stmts.insert(stmts.begin() + static_cast<std::ptrdiff_t>(i), moved.begin(), moved.end());
            i += moved.size();
        }
    }

    const Bindings& b_;
    int hoisted_{0};
};

} // namespace

void substitute(Body& stmts, const std::string& name, const ExprPtr& value) {
    auto f = [&](const ExprPtr& x) { return x->kind == ExprKind::Var && x->name == name ? value : x; };
    transformBody(stmts, f);
}

int removeLets(Body& stmts, const std::string& name) {
    int removed = 0;
    for (size_t i = 0; i < stmts.size();) {
        if (stmts[i]->kind == StmtKind::Let && stmts[i]->name == name) {
            stmts.erase(stmts.begin() + static_cast<std::ptrdiff_t>(i));
            ++removed;
            continue;
        }
        removed += removeLets(stmts[i]->body, name);
        removed += removeLets(stmts[i]->elseBody, name);
        ++i;
    }
    return removed;
}

SimplifyReport simplify(Kernel& kernel) {
    SimplifyReport report;
    report.propagated = propagateConstants(kernel);
    Bindings b;
    collectBindings(kernel.body, b);
    report.folded = DivRemFolder(b.assigned).run(kernel.body);
    report.hoisted = Hoister(b).run(kernel.body);
    return report;
}

} // namespace dlc::loopir
