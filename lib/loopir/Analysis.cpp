#include "dlc/loopir/Analysis.h"
#include "dlc/ir/Error.h"
#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <vector>

namespace dlc::loopir {

namespace {

constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

int64_t satAdd(int64_t a, int64_t b) {
    int64_t r;
    if (__builtin_add_overflow(a, b, &r)) return (a > 0) ? kMax : kMin;
    return r;
}

int64_t satMul(int64_t a, int64_t b) {
    int64_t r;
    if (__builtin_mul_overflow(a, b, &r)) return ((a > 0) == (b > 0)) ? kMax : kMin;
    return r;
}

} // namespace

Interval Interval::unknown() { return {kMin, kMax}; }
bool Interval::known() const { return lo != kMin && hi != kMax; }

std::optional<Interval> RangeEnv::get(const std::string& name) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) return found->second;
    }
    return std::nullopt;
}

Interval rangeOf(const ExprPtr& e, const RangeEnv& env) {
    switch (e->kind) {
        case ExprKind::Int: return {e->intValue, e->intValue};
        case ExprKind::Var: return env.get(e->name).value_or(Interval::unknown());
        case ExprKind::Cast: return e->type == ScalarType::F32 ? Interval::unknown() : rangeOf(e->args[0], env);
        case ExprKind::Select: {
            Interval a = rangeOf(e->args[1], env);
            Interval b = rangeOf(e->args[2], env);
            return {std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
        }
        case ExprKind::Binary: {
            if (e->type != ScalarType::I64 && e->type != ScalarType::I32) return Interval::unknown();
            Interval a = rangeOf(e->args[0], env);
            Interval b = rangeOf(e->args[1], env);
            switch (e->binary) {
                case BinaryOp::Add: return {satAdd(a.lo, b.lo), satAdd(a.hi, b.hi)};
                case BinaryOp::Sub: return {satAdd(a.lo, -b.hi), satAdd(a.hi, -b.lo)};
                case BinaryOp::Mul: {
                    int64_t c[] = {satMul(a.lo, b.lo), satMul(a.lo, b.hi), satMul(a.hi, b.lo), satMul(a.hi, b.hi)};
                    return {*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
                }
                case BinaryOp::Div:
                    if (b.lo == b.hi && b.lo > 0 && a.lo >= 0) return {a.lo / b.lo, a.hi == kMax ? kMax : a.hi / b.lo};
                    return Interval::unknown();
                case BinaryOp::Rem:
                    if (b.lo == b.hi && b.lo > 0 && a.lo >= 0) {
                        if (a.hi != kMax && a.hi / b.lo == a.lo / b.lo) return {a.lo % b.lo, a.hi % b.lo};
                        return {0, b.lo - 1};
                    }
                    return Interval::unknown();
                case BinaryOp::Min: return {std::min(a.lo, b.lo), std::min(a.hi, b.hi)};
                case BinaryOp::Max:
                case BinaryOp::MaxNaN: return {std::max(a.lo, b.lo), std::max(a.hi, b.hi)};
                default: return Interval::unknown();
            }
        }
        default: return Interval::unknown();
    }
}

std::optional<Affine> affineForm(const ExprPtr& e) {
    switch (e->kind) {
        case ExprKind::Int: return Affine{{}, e->intValue};
        case ExprKind::Var: return Affine{{{e->name, 1}}, 0};
        case ExprKind::Cast: return e->type == ScalarType::F32 ? std::nullopt : affineForm(e->args[0]);
        case ExprKind::Binary: {
            auto a = affineForm(e->args[0]);
            auto b = affineForm(e->args[1]);
            if (!a || !b) return std::nullopt;
            if (e->binary == BinaryOp::Add || e->binary == BinaryOp::Sub) {
                int64_t sign = e->binary == BinaryOp::Add ? 1 : -1;
                for (const auto& [v, c] : b->coeffs) a->coeffs[v] += sign * c;
                a->constant += sign * b->constant;
                return a;
            }
            if (e->binary == BinaryOp::Mul) {
                const Affine* scalar = a->coeffs.empty() ? &*a : (b->coeffs.empty() ? &*b : nullptr);
                if (!scalar) return std::nullopt;
                const Affine& other = scalar == &*a ? *b : *a;
                Affine out{{}, other.constant * scalar->constant};
                for (const auto& [v, c] : other.coeffs) out.coeffs[v] = c * scalar->constant;
                return out;
            }
            return std::nullopt;
        }
        default: return std::nullopt;
    }
}

namespace {

bool dependsOnData(const ExprPtr& e, const std::unordered_set<std::string>& dataLocals) {
    if (e->kind == ExprKind::Load) return true;
    if (e->kind == ExprKind::Var && dataLocals.count(e->name)) return true;
    return std::any_of(e->args.begin(), e->args.end(), [&](const ExprPtr& a) { return dependsOnData(a, dataLocals); });
}

// Narrows the range of `var OP constant` (either side) on the branch that `taken` selects. Returns false if the
// branch cannot run (the range becomes empty).
bool refine(RangeEnv& env, const ExprPtr& cond, bool taken) {
    if (cond->kind != ExprKind::Compare || cond->args[0]->type == ScalarType::F32) return true;
    ExprPtr lhs = cond->args[0];
    ExprPtr rhs = cond->args[1];
    CompareOp op = cond->compare;
    if (lhs->kind == ExprKind::Int && rhs->kind == ExprKind::Var) { // c OP x  →  x OP' c
        std::swap(lhs, rhs);
        switch (op) {
            case CompareOp::LT: op = CompareOp::GT; break;
            case CompareOp::LE: op = CompareOp::GE; break;
            case CompareOp::GT: op = CompareOp::LT; break;
            case CompareOp::GE: op = CompareOp::LE; break;
            default: break;
        }
    }
    if (lhs->kind != ExprKind::Var || rhs->kind != ExprKind::Int) return true;
    if (!taken) { // the else branch: the negated comparison
        switch (op) {
            case CompareOp::LT: op = CompareOp::GE; break;
            case CompareOp::LE: op = CompareOp::GT; break;
            case CompareOp::GT: op = CompareOp::LE; break;
            case CompareOp::GE: op = CompareOp::LT; break;
            case CompareOp::EQ: op = CompareOp::NE; break;
            case CompareOp::NE: op = CompareOp::EQ; break;
        }
    }
    Interval r = env.get(lhs->name).value_or(Interval::unknown());
    int64_t c = rhs->intValue;
    switch (op) {
        case CompareOp::LT: r.hi = std::min(r.hi, c - 1); break;
        case CompareOp::LE: r.hi = std::min(r.hi, c); break;
        case CompareOp::GT: r.lo = std::max(r.lo, c + 1); break;
        case CompareOp::GE: r.lo = std::max(r.lo, c); break;
        case CompareOp::EQ: r = {std::max(r.lo, c), std::min(r.hi, c)}; break;
        case CompareOp::NE: return true;
    }
    env.set(lhs->name, r);
    return r.lo <= r.hi;
}

ExprPtr zeroOf(ScalarType t) { return t == ScalarType::F32 ? floatConst(0.0) : cast(t, intConst(0)); }

ExprPtr inBounds(const ExprPtr& index, int64_t n, ExprPtr ifIn, ExprPtr ifOut) {
    return select(compare(CompareOp::GE, index, intConst(0)),
                  select(compare(CompareOp::LT, index, intConst(n)), std::move(ifIn), ifOut), ifOut);
}

// Decides a comparison of integer ranges when the ranges don't overlap: 1 = always true, 0 = always false.
std::optional<bool> decide(const ExprPtr& cond, const RangeEnv& env) {
    if (cond->kind != ExprKind::Compare || cond->args[0]->type == ScalarType::F32) return std::nullopt;
    Interval a = rangeOf(cond->args[0], env);
    Interval b = rangeOf(cond->args[1], env);
    if (!a.known() || !b.known()) return std::nullopt;
    switch (cond->compare) {
        case CompareOp::LT: if (a.hi < b.lo) return true; if (a.lo >= b.hi) return false; break;
        case CompareOp::LE: if (a.hi <= b.lo) return true; if (a.lo > b.hi) return false; break;
        case CompareOp::GT: if (a.lo > b.hi) return true; if (a.hi <= b.lo) return false; break;
        case CompareOp::GE: if (a.lo >= b.hi) return true; if (a.hi < b.lo) return false; break;
        case CompareOp::EQ: if (a.lo == a.hi && b.lo == b.hi && a.lo == b.lo) return true; if (a.hi < b.lo || b.hi < a.lo) return false; break;
        case CompareOp::NE: if (a.hi < b.lo || b.hi < a.lo) return true; if (a.lo == a.hi && b.lo == b.hi && a.lo == b.lo) return false; break;
    }
    return std::nullopt;
}

class BoundsChecker {
public:
    BoundsChecker(Kernel& k, BoundsReport& report) : k_(k), report_(report) {}

    void run() { body(k_.body); }

private:
    ExprPtr rewrite(const ExprPtr& e) {
        if (e->kind != ExprKind::Load && e->args.empty()) return e;
        std::vector<ExprPtr> args;
        bool changed = false;
        for (const auto& a : e->args) {
            args.push_back(rewrite(a));
            changed = changed || args.back() != a;
        }
        if (e->kind == ExprKind::Load) {
            ++report_.accesses;
            int64_t n = k_.buffers[e->buffer].elements();
            Interval r = rangeOf(args[0], env_);
            if (r.within(0, n - 1)) {
                ++report_.proven;
            } else if (dependsOnData(args[0], dataLocals_)) {
                ++report_.guarded;
                ExprPtr clamped = binary(BinaryOp::Max, binary(BinaryOp::Min, args[0], intConst(n - 1)), intConst(0));
                return inBounds(args[0], n, load(e->buffer, clamped, e->type), zeroOf(e->type));
            } else {
                throw IRError("bounds check: " + k_.name + " reads " + k_.buffers[e->buffer].name + "[" +
                              toString(args[0], k_) + "] with index range [" + std::to_string(r.lo) + ", " +
                              std::to_string(r.hi) + "], outside its " + std::to_string(n) + " elements");
            }
        }
        if (!changed) return e;
        auto copy = std::make_shared<Expr>(*e);
        copy->args = std::move(args);
        return copy;
    }

    void body(Body& stmts) {
        env_.push();
        for (size_t i = 0; i < stmts.size(); ++i) {
            StmtPtr& s = stmts[i];
            switch (s->kind) {
                case StmtKind::For:
                    if (s->extent <= 0) break; // never runs
                    env_.push();
                    env_.set(s->var, {0, s->extent - 1});
                    body(s->body);
                    env_.pop();
                    break;
                case StmtKind::Let:
                    s->value = rewrite(s->value);
                    if (s->type != ScalarType::F32) env_.set(s->name, rangeOf(s->value, env_));
                    if (dependsOnData(s->value, dataLocals_)) dataLocals_.insert(s->name);
                    break;
                case StmtKind::Assign: s->value = rewrite(s->value); break;
                case StmtKind::Store:
                case StmtKind::AtomicAdd: {
                    s->value = rewrite(s->value);
                    s->index = rewrite(s->index);
                    ++report_.accesses;
                    int64_t n = k_.buffers[s->buffer].elements();
                    Interval r = rangeOf(s->index, env_);
                    if (r.within(0, n - 1)) {
                        ++report_.proven;
                    } else if (dependsOnData(s->index, dataLocals_)) {
                        ++report_.guarded;
                        StmtPtr guarded = ifThen(compare(CompareOp::GE, s->index, intConst(0)),
                                                 {ifThen(compare(CompareOp::LT, s->index, intConst(n)), {s})});
                        stmts[i] = guarded;
                    } else {
                        throw IRError("bounds check: " + k_.name + " writes " + k_.buffers[s->buffer].name + "[" +
                                      toString(s->index, k_) + "] with index range [" + std::to_string(r.lo) + ", " +
                                      std::to_string(r.hi) + "], outside its " + std::to_string(n) + " elements");
                    }
                    break;
                }
                case StmtKind::If:
                    s->cond = rewrite(s->cond);
                    for (bool taken : {true, false}) {
                        env_.push();
                        if (refine(env_, s->cond, taken)) body(taken ? s->body : s->elseBody);
                        env_.pop();
                    }
                    break;
                case StmtKind::Barrier: break;
            }
        }
        env_.pop();
    }

    Kernel& k_;
    BoundsReport& report_;
    RangeEnv env_;
    std::unordered_set<std::string> dataLocals_;
};

// Replaces every `if` whose condition is decided by range analysis with the branch it takes.
int eliminate(Body& stmts, RangeEnv& env) {
    int removed = 0;
    env.push();
    for (size_t i = 0; i < stmts.size(); ++i) {
        Stmt& s = *stmts[i];
        if (s.kind == StmtKind::For) {
            if (s.extent <= 0) continue;
            env.push();
            env.set(s.var, {0, s.extent - 1});
            removed += eliminate(s.body, env);
            env.pop();
        } else if (s.kind == StmtKind::Let && s.type != ScalarType::F32) {
            env.set(s.name, rangeOf(s.value, env));
        } else if (s.kind == StmtKind::If) {
            if (auto taken = decide(s.cond, env)) {
                Body branch = *taken ? s.body : s.elseBody;
                stmts.erase(stmts.begin() + static_cast<std::ptrdiff_t>(i));
                stmts.insert(stmts.begin() + static_cast<std::ptrdiff_t>(i), branch.begin(), branch.end());
                ++removed;
                --i; // revisit the spliced statements
                continue;
            }
            for (bool taken : {true, false}) {
                env.push();
                if (refine(env, s.cond, taken)) removed += eliminate(taken ? s.body : s.elseBody, env);
                env.pop();
            }
        }
    }
    env.pop();
    return removed;
}

// --- Schedule verification -----------------------------------------------------------------------------------------

struct Footprint {
    std::map<int, Interval> reads, writes;
};

void accumulate(std::map<int, Interval>& m, int buffer, Interval r) {
    auto [it, inserted] = m.try_emplace(buffer, r);
    if (!inserted) it->second = {std::min(it->second.lo, r.lo), std::max(it->second.hi, r.hi)};
}

class FootprintWalker {
public:
    FootprintWalker(const Kernel& k, RangeEnv& env) : k_(k), env_(env) {}

    Footprint walk(const Body& stmts) {
        Footprint f;
        body(stmts, f);
        return f;
    }

private:
    void loads(const ExprPtr& e, Footprint& f) {
        for (const auto& a : e->args) loads(a, f);
        if (e->kind == ExprKind::Load && k_.buffers[e->buffer].scope == MemScope::Global) {
            accumulate(f.reads, e->buffer, rangeOf(e->args[0], env_));
        }
    }

    void body(const Body& stmts, Footprint& f) {
        env_.push();
        for (const auto& s : stmts) {
            switch (s->kind) {
                case StmtKind::For:
                    if (s->extent <= 0) break;
                    env_.push();
                    env_.set(s->var, {0, s->extent - 1});
                    body(s->body, f);
                    env_.pop();
                    break;
                case StmtKind::Let:
                    loads(s->value, f);
                    if (s->type != ScalarType::F32) env_.set(s->name, rangeOf(s->value, env_));
                    break;
                case StmtKind::Assign: loads(s->value, f); break;
                case StmtKind::Store:
                case StmtKind::AtomicAdd:
                    loads(s->value, f);
                    loads(s->index, f);
                    if (k_.buffers[s->buffer].scope == MemScope::Global) {
                        accumulate(f.writes, s->buffer, rangeOf(s->index, env_));
                    }
                    break;
                case StmtKind::If:
                    loads(s->cond, f);
                    for (bool taken : {true, false}) {
                        env_.push();
                        if (refine(env_, s->cond, taken)) body(taken ? s->body : s->elseBody, f);
                        env_.pop();
                    }
                    break;
                case StmtKind::Barrier: break;
            }
        }
        env_.pop();
    }

    const Kernel& k_;
    RangeEnv& env_;
};

constexpr int64_t kMaxEnumerated = 1 << 16;

void verifyParallel(const Kernel& k, const Stmt& loop, RangeEnv& env) {
    if (loop.extent > kMaxEnumerated) return; // beyond what is enumerated (no schedule makes such loops today)
    std::vector<Footprint> iterations;
    for (int64_t v = 0; v < loop.extent; ++v) {
        env.push();
        env.set(loop.var, {v, v});
        FootprintWalker walker(k, env);
        iterations.push_back(walker.walk(loop.body));
        env.pop();
    }
    std::set<int> written;
    for (const auto& it : iterations) {
        for (const auto& [b, _] : it.writes) written.insert(b);
    }
    for (int b : written) {
        std::vector<std::pair<Interval, int64_t>> ranges;
        for (int64_t v = 0; v < loop.extent; ++v) {
            auto w = iterations[v].writes.find(b);
            if (w == iterations[v].writes.end()) continue;
            if (!w->second.known()) {
                // Data-dependent writes (a guarded scatter) must not run in parallel.
                throw IRError("schedule: " + k.name + " writes " + k.buffers[b].name + " at a data-dependent index in "
                              "parallel loop " + loop.var);
            }
            ranges.emplace_back(w->second, v);
        }
        std::sort(ranges.begin(), ranges.end(), [](const auto& x, const auto& y) { return x.first.lo < y.first.lo; });
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first.lo <= ranges[i - 1].first.hi) {
                throw IRError("schedule: iterations " + std::to_string(ranges[i - 1].second) + " and " +
                              std::to_string(ranges[i].second) + " of parallel loop " + loop.var + " in " + k.name +
                              " may write the same elements of " + k.buffers[b].name);
            }
        }
        // Each iteration reads only what it writes itself, or what no iteration writes.
        for (int64_t v = 0; v < loop.extent; ++v) {
            auto r = iterations[v].reads.find(b);
            if (r == iterations[v].reads.end()) continue;
            for (const auto& [w, other] : ranges) {
                if (other == v) continue;
                if (r->second.lo <= w.hi && w.lo <= r->second.hi) {
                    throw IRError("schedule: iteration " + std::to_string(v) + " of parallel loop " + loop.var + " in " +
                                  k.name + " reads elements of " + k.buffers[b].name + " that iteration " +
                                  std::to_string(other) + " writes");
                }
            }
        }
    }
}

bool mentions(const ExprPtr& e, const std::string& name) {
    if (e->kind == ExprKind::Var && e->name == name) return true;
    return std::any_of(e->args.begin(), e->args.end(), [&](const ExprPtr& a) { return mentions(a, name); });
}

void collectStores(const Body& stmts, std::vector<const Stmt*>& out) {
    for (const auto& s : stmts) {
        if (s->kind == StmtKind::Store || s->kind == StmtKind::AtomicAdd) out.push_back(s.get());
        collectStores(s->body, out);
        collectStores(s->elseBody, out);
    }
}

void collectLoads(const ExprPtr& e, std::vector<const Expr*>& out) {
    if (e->kind == ExprKind::Load) out.push_back(e.get());
    for (const auto& a : e->args) collectLoads(a, out);
}

void collectLoads(const Body& stmts, std::vector<const Expr*>& out) {
    for (const auto& s : stmts) {
        for (const ExprPtr* e : {&s->value, &s->index, &s->cond}) {
            if (*e) collectLoads(*e, out);
        }
        collectLoads(s->body, out);
        collectLoads(s->elseBody, out);
    }
}

void verifyVectorized(const Kernel& k, const Stmt& loop) {
    std::vector<const Stmt*> stores;
    collectStores(loop.body, stores);
    std::vector<const Expr*> loads;
    collectLoads(loop.body, loads);
    for (const Stmt* s : stores) {
        const BufferDecl& b = k.buffers[s->buffer];
        if (b.scope != MemScope::Global) continue; // a register accumulator: a reduction
        auto affine = affineForm(s->index);
        bool varies = affine ? affine->coeffs.count(loop.var) && affine->coeffs.at(loop.var) != 0
                             : mentions(s->index, loop.var);
        if (!varies) {
            throw IRError("schedule: every iteration of vectorized loop " + loop.var + " in " + k.name +
                          " writes the same element of " + b.name);
        }
        std::string written = toString(s->index, k);
        for (const Expr* l : loads) {
            if (l->buffer == s->buffer && toString(l->args[0], k) != written) {
                throw IRError("schedule: vectorized loop " + loop.var + " in " + k.name + " reads " + b.name + "[" +
                              toString(l->args[0], k) + "] while writing " + b.name + "[" + written + "]");
            }
        }
    }
}

void verifyLoops(const Kernel& k, const Body& stmts, RangeEnv& env) {
    env.push();
    for (const auto& s : stmts) {
        if (s->kind == StmtKind::For) {
            if (s->extent <= 0) continue;
            if (s->loop == LoopKind::Parallel) verifyParallel(k, *s, env);
            if (s->loop == LoopKind::Vectorized) verifyVectorized(k, *s);
            env.push();
            env.set(s->var, {0, s->extent - 1});
            verifyLoops(k, s->body, env);
            env.pop();
        } else if (s->kind == StmtKind::Let && s->type != ScalarType::F32) {
            env.set(s->name, rangeOf(s->value, env));
        } else if (s->kind == StmtKind::If) {
            for (bool taken : {true, false}) {
                env.push();
                if (refine(env, s->cond, taken)) verifyLoops(k, taken ? s->body : s->elseBody, env);
                env.pop();
            }
        }
    }
    env.pop();
}

} // namespace

BoundsReport checkBounds(Kernel& kernel) {
    BoundsReport report;
    BoundsChecker(kernel, report).run();
    RangeEnv env;
    report.eliminated = eliminate(kernel.body, env);
    return report;
}

void verifySchedule(const Kernel& kernel) {
    RangeEnv env;
    verifyLoops(kernel, kernel.body, env);
}

} // namespace dlc::loopir
