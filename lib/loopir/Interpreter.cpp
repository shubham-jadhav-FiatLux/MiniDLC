#include "dlc/loopir/Interpreter.h"
#include "dlc/ir/Error.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace dlc::loopir {

namespace {

struct Scalar {
    ScalarType type{ScalarType::I64};
    float f{0.0f};
    int64_t i{0}; // I32, I64 and Bool values
};

Scalar makeF(float v) { return {ScalarType::F32, v, 0}; }
Scalar makeI(ScalarType t, int64_t v) { return {t, 0.0f, v}; }

float maxNaN(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<float>::quiet_NaN();
    if (a == 0.0f && b == 0.0f) return std::signbit(a) ? b : a; // max(-0, +0) = +0
    return a > b ? a : b;
}

template <class Ctx> Scalar evaluate(const Expr& e, Ctx& ctx);

template <class Ctx> Scalar evalBinary(const Expr& e, Ctx& ctx);

// Evaluates `e`; the context supplies variables, loads and shuffles.
template <class Ctx> Scalar evaluate(const Expr& e, Ctx& ctx) {
    switch (e.kind) {
        case ExprKind::Int: return makeI(ScalarType::I64, e.intValue);
        case ExprKind::Float: return makeF(static_cast<float>(e.floatValue));
        case ExprKind::Var: return ctx.variable(e);
        case ExprKind::Load: return ctx.load(e, evaluate(*e.args[0], ctx).i);
        case ExprKind::Binary: return evalBinary(e, ctx);
        case ExprKind::Unary: {
            Scalar a = evaluate(*e.args[0], ctx);
            if (a.type != ScalarType::F32) {
                if (e.unary == UnaryOp::Neg) return makeI(a.type, -a.i);
                throw IRError("interpreter: integer " + std::to_string(static_cast<int>(e.unary)));
            }
            switch (e.unary) {
                case UnaryOp::Neg: return makeF(-a.f);
                case UnaryOp::Exp: return makeF(std::exp(a.f));
                case UnaryOp::Log: return makeF(std::log(a.f));
                case UnaryOp::Tanh: return makeF(std::tanh(a.f));
                case UnaryOp::Sqrt: return makeF(std::sqrt(a.f));
                case UnaryOp::Rsqrt: return makeF(1.0f / std::sqrt(a.f));
            }
            break;
        }
        case ExprKind::Select: return evaluate(*e.args[0], ctx).i ? evaluate(*e.args[1], ctx) : evaluate(*e.args[2], ctx);
        case ExprKind::Compare: {
            Scalar a = evaluate(*e.args[0], ctx);
            Scalar b = evaluate(*e.args[1], ctx);
            bool r = false;
            if (a.type == ScalarType::F32) {
                switch (e.compare) {
                    case CompareOp::LT: r = a.f < b.f; break;
                    case CompareOp::LE: r = a.f <= b.f; break;
                    case CompareOp::GT: r = a.f > b.f; break;
                    case CompareOp::GE: r = a.f >= b.f; break;
                    case CompareOp::EQ: r = a.f == b.f; break;
                    case CompareOp::NE: r = a.f != b.f; break;
                }
            } else {
                switch (e.compare) {
                    case CompareOp::LT: r = a.i < b.i; break;
                    case CompareOp::LE: r = a.i <= b.i; break;
                    case CompareOp::GT: r = a.i > b.i; break;
                    case CompareOp::GE: r = a.i >= b.i; break;
                    case CompareOp::EQ: r = a.i == b.i; break;
                    case CompareOp::NE: r = a.i != b.i; break;
                }
            }
            return makeI(ScalarType::Bool, r);
        }
        case ExprKind::Cast: {
            Scalar a = evaluate(*e.args[0], ctx);
            if (e.type == ScalarType::F32) return makeF(a.type == ScalarType::F32 ? a.f : static_cast<float>(a.i));
            if (a.type == ScalarType::F32) return makeI(e.type, static_cast<int64_t>(std::trunc(a.f)));
            return makeI(e.type, e.type == ScalarType::I32 ? static_cast<int32_t>(a.i) : a.i);
        }
        case ExprKind::Shuffle: return ctx.shuffle(e);
    }
    throw IRError("interpreter: unknown expression");
}

template <class Ctx> Scalar evalBinary(const Expr& e, Ctx& ctx) {
    Scalar a = evaluate(*e.args[0], ctx);
    Scalar b = evaluate(*e.args[1], ctx);
    if (e.binary == BinaryOp::Fma) return makeF(std::fma(a.f, b.f, evaluate(*e.args[2], ctx).f));
    if (a.type == ScalarType::F32) {
        switch (e.binary) {
            case BinaryOp::Add: return makeF(a.f + b.f);
            case BinaryOp::Sub: return makeF(a.f - b.f);
            case BinaryOp::Mul: return makeF(a.f * b.f);
            case BinaryOp::Div: return makeF(a.f / b.f);
            case BinaryOp::Min: return makeF(std::fmin(a.f, b.f));
            case BinaryOp::Max: return makeF(std::fmax(a.f, b.f));
            case BinaryOp::MaxNaN: return makeF(maxNaN(a.f, b.f));
            default: break;
        }
        throw IRError("interpreter: bad float op");
    }
    switch (e.binary) {
        case BinaryOp::Add: return makeI(a.type, a.i + b.i);
        case BinaryOp::Sub: return makeI(a.type, a.i - b.i);
        case BinaryOp::Mul: return makeI(a.type, a.i * b.i);
        case BinaryOp::Div:
            if (b.i == 0) throw IRError("interpreter: integer division by zero");
            return makeI(a.type, a.i / b.i);
        case BinaryOp::Rem:
            if (b.i == 0) throw IRError("interpreter: integer remainder by zero");
            return makeI(a.type, a.i % b.i);
        case BinaryOp::Min: return makeI(a.type, std::min(a.i, b.i));
        case BinaryOp::Max:
        case BinaryOp::MaxNaN: return makeI(a.type, std::max(a.i, b.i));
        default: break;
    }
    throw IRError("interpreter: bad integer op");
}


class Interpreter {
public:
    Interpreter(const Kernel& kernel, const std::vector<void*>& args) : kernel_(kernel) {
        auto globals = kernel.globalBuffers();
        if (globals.size() != args.size()) throw IRError("interpreter: wrong number of buffers");
        memory_.resize(kernel.buffers.size(), nullptr);
        for (size_t g = 0; g < globals.size(); ++g) memory_[globals[g]] = static_cast<char*>(args[g]);
        for (size_t b = 0; b < kernel.buffers.size(); ++b) {
            if (kernel.buffers[b].scope == MemScope::Global) continue;
            locals_.emplace_back(kernel.buffers[b].elements() * 4, 0);
            memory_[b] = locals_.back().data();
        }
    }

    void run(int64_t begin, int64_t end) {
        scopes_.emplace_back();
        if (kernel_.parallelExtent() > 1 || (kernel_.body.size() == 1 && kernel_.body[0]->kind == StmtKind::For &&
                                              kernel_.body[0]->loop == LoopKind::Parallel)) {
            const Stmt& loop = *kernel_.body[0];
            for (int64_t v = begin; v < std::min(end, loop.extent); ++v) iteration(loop, v);
        } else {
            body(kernel_.body);
        }
        scopes_.pop_back();
    }

private:
    void iteration(const Stmt& loop, int64_t v) {
        scopes_.emplace_back();
        scopes_.back()[loop.var] = makeI(ScalarType::I64, v);
        body(loop.body);
        scopes_.pop_back();
    }

    void body(const Body& stmts) {
        for (const auto& s : stmts) stmt(*s);
    }

    void stmt(const Stmt& s) {
        switch (s.kind) {
            case StmtKind::For:
                for (int64_t v = 0; v < s.extent; ++v) iteration(s, v);
                break;
            case StmtKind::Let: scopes_.back()[s.name] = eval(*s.value); break;
            case StmtKind::Assign: lookup(s.name) = eval(*s.value); break;
            case StmtKind::Store: write(s.buffer, eval(*s.index).i, eval(*s.value), false); break;
            case StmtKind::AtomicAdd: write(s.buffer, eval(*s.index).i, eval(*s.value), true); break;
            case StmtKind::If: {
                scopes_.emplace_back();
                body(eval(*s.cond).i ? s.body : s.elseBody);
                scopes_.pop_back();
                break;
            }
            case StmtKind::Barrier: break;
        }
    }

public:
    Scalar variable(const Expr& e) { return lookup(e.name); }
    Scalar load(const Expr& e, int64_t index) { return read(e.buffer, index); }
    Scalar shuffle(const Expr&) { throw IRError("interpreter: warp shuffles run only in GPU kernels"); }
    Scalar eval(const Expr& e) { return evaluate(e, *this); }

private:
    Scalar& lookup(const std::string& name) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second;
        }
        throw IRError("interpreter: undefined variable '" + name + "' in " + kernel_.name);
    }

    char* address(int buffer, int64_t index) {
        const BufferDecl& b = kernel_.buffers[buffer];
        if (index < 0 || index >= b.elements()) {
            throw IRError("interpreter: " + kernel_.name + " accesses " + b.name + "[" + std::to_string(index) +
                          "], outside its " + std::to_string(b.elements()) + " elements");
        }
        return memory_[buffer] + index * 4;
    }

public:
    Scalar read(int buffer, int64_t index) {
        char* p = address(buffer, index);
        if (kernel_.buffers[buffer].type == ScalarType::F32) {
            float v;
            std::memcpy(&v, p, 4);
            return makeF(v);
        }
        int32_t v;
        std::memcpy(&v, p, 4);
        return makeI(ScalarType::I32, v);
    }

private:
    void write(int buffer, int64_t index, const Scalar& value, bool accumulate) {
        char* p = address(buffer, index);
        if (kernel_.buffers[buffer].type == ScalarType::F32) {
            float v = value.f;
            if (accumulate) {
                float old;
                std::memcpy(&old, p, 4);
                v += old;
            }
            std::memcpy(p, &v, 4);
        } else {
            int32_t v = static_cast<int32_t>(value.i);
            std::memcpy(p, &v, 4);
        }
    }

    const Kernel& kernel_;
    std::vector<char*> memory_;
    std::vector<std::vector<char>> locals_;
    std::vector<std::unordered_map<std::string, Scalar>> scopes_;
};

// Runs a GPU kernel block by block. The threads of a block execute in lockstep: every statement runs for all active
// threads before the next one, an If splits the active set, a barrier must be reached by every thread of the block
// (anything else is a divergent barrier, an error), and a shuffle reads the source lane's value, which must be active.
// Shared memory starts as NaN, so a read before the write that should precede it shows up in the results.
class GpuInterpreter {
public:
    GpuInterpreter(const Kernel& kernel, const std::vector<void*>& args, GpuLaunchCheck* launch) : kernel_(kernel) {
        if (!launch) {
            ownLaunch_ = std::make_unique<GpuLaunchCheck>(kernel);
            launch = ownLaunch_.get();
        }
        launch_ = launch;
        auto globals = kernel.globalBuffers();
        if (globals.size() != args.size()) throw IRError("interpreter: wrong number of buffers");
        memory_.resize(kernel.buffers.size(), nullptr);
        for (size_t g = 0; g < globals.size(); ++g) memory_[globals[g]] = static_cast<char*>(args[g]);
        const Body* b = &kernel.body;
        while (b->size() == 1 && (*b)[0]->kind == StmtKind::For && (*b)[0]->loop == LoopKind::Bound) {
            const Stmt& loop = *(*b)[0];
            int axis = static_cast<int>(loop.axis);
            boundVars_[axis] = id(loop.var);
            int64_t expected = axis < 3 ? kernel.grid[axis] : kernel.block[axis - 3];
            if (loop.extent != expected) throw IRError("interpreter: " + kernel.name + ": bound loop " + loop.var + " disagrees with the launch configuration");
            b = &loop.body;
        }
        threadBody_ = b;
        threads_ = kernel.block[0] * kernel.block[1] * kernel.block[2];
        sharedSlot_.assign(kernel.buffers.size(), -1);
        for (size_t i = 0; i < kernel.buffers.size(); ++i) {
            const BufferDecl& d = kernel.buffers[i];
            if (d.scope == MemScope::Shared) {
                sharedSlot_[i] = static_cast<int>(shared_.size());
                shared_.emplace_back(i, std::vector<float>(d.elements()));
                sharedAccess_.emplace_back(d.elements());
            }
            if (d.scope == MemScope::Register) registers_.emplace_back(i, std::vector<float>(d.elements() * threads_));
        }
        collect(*threadBody_);
        env_.assign(threads_ * ids_.size(), Scalar{});
        active_.assign(threads_, 0);
    }

    void run(int64_t begin, int64_t end) {
        if (kernel_.body.empty()) return;
        const int64_t blocks = kernel_.grid[0] * kernel_.grid[1] * kernel_.grid[2];
        for (int64_t b = begin; b < std::min(end, blocks); ++b) block(b);
    }

    Scalar variable(const Expr& e) { return env_[current_ * ids_.size() + id(e.name)]; }
    Scalar load(const Expr& e, int64_t index) { return read(e.buffer, index); }

    Scalar read(int buffer, int64_t index) {
        char* p = address(buffer, index, current_);
        track(buffer, index, current_, Access::Read);
        if (kernel_.buffers[buffer].type == ScalarType::F32) {
            float v;
            std::memcpy(&v, p, 4);
            return makeF(v);
        }
        int32_t v;
        std::memcpy(&v, p, 4);
        return makeI(ScalarType::I32, v);
    }

    Scalar shuffle(const Expr& e) {
        const int64_t lane = current_ % 32;
        const int64_t m = evaluate(*e.args[1], *this).i;
        int64_t src = e.butterfly ? (lane ^ m) : lane + m;
        if (src >= 32 || current_ - lane + src >= threads_) {
            if (e.butterfly) throw IRError("interpreter: " + kernel_.name + ": butterfly shuffle past the warp");
            src = lane; // shfl.down keeps the lane's own value past the end
        }
        const int64_t thread = current_ - lane + src;
        if (!active_[thread]) throw IRError("interpreter: " + kernel_.name + ": shuffle reads inactive lane " + std::to_string(src));
        const int64_t self = current_;
        current_ = thread;
        Scalar v = evaluate(*e.args[0], *this);
        current_ = self;
        return v;
    }

private:
    void collect(const Body& body) {
        for (const auto& s : body) {
            if (s->kind == StmtKind::For) id(s->var);
            if (s->kind == StmtKind::Let) id(s->name);
            collect(s->body);
            collect(s->elseBody);
        }
    }

    size_t id(const std::string& name) {
        auto [it, inserted] = ids_.try_emplace(name, ids_.size());
        if (inserted && !env_.empty()) throw IRError("interpreter: " + kernel_.name + ": undefined variable '" + name + "'");
        return it->second;
    }

    void block(int64_t b) {
        const auto& g = kernel_.grid;
        const int64_t coords[3] = {b % g[0], (b / g[0]) % g[1], b / (g[0] * g[1])};
        for (auto& [i, mem] : shared_) {
            std::fill(mem.begin(), mem.end(), std::numeric_limits<float>::quiet_NaN());
            memory_[i] = reinterpret_cast<char*>(mem.data());
        }
        for (auto& accesses : sharedAccess_) std::fill(accesses.begin(), accesses.end(), SharedAccess{});
        epoch_ = 0;
        block_ = b;
        globalAccess_.clear();
        std::vector<int64_t> all(threads_);
        const auto& bl = kernel_.block;
        for (int64_t t = 0; t < threads_; ++t) {
            all[t] = t;
            const int64_t tc[3] = {t % bl[0], (t / bl[0]) % bl[1], t / (bl[0] * bl[1])};
            for (int a = 0; a < 6; ++a) {
                if (boundVars_[a] < 0) continue;
                env_[t * ids_.size() + boundVars_[a]] = makeI(ScalarType::I64, a < 3 ? coords[a] : tc[a - 3]);
            }
        }
        body(*threadBody_, all);
        // The block's accesses to global memory, against the other blocks of the launch.
        for (const auto& [key, a] : globalAccess_) {
            uint8_t flags = (a.reader != kNone ? GpuLaunchCheck::kRead : 0) | (a.writer != kNone ? GpuLaunchCheck::kWrite : 0) |
                            (a.adder != kNone ? GpuLaunchCheck::kAtomic : 0);
            launch_->merge(static_cast<int>(key >> 40), static_cast<int64_t>(key & ((uint64_t{1} << 40) - 1)), flags, b);
        }
    }

    void body(const Body& stmts, const std::vector<int64_t>& active) {
        if (active.empty()) return;
        for (int64_t t : active) active_[t] = 1;
        for (const auto& s : stmts) stmt(*s, active);
        for (int64_t t : active) active_[t] = 0;
    }

    void setAll(size_t var, const std::vector<int64_t>& active, const std::vector<Scalar>& values) {
        for (size_t k = 0; k < active.size(); ++k) env_[active[k] * ids_.size() + var] = values[k];
    }

    std::vector<Scalar> evalAll(const Expr& e, const std::vector<int64_t>& active) {
        std::vector<Scalar> values(active.size());
        for (size_t k = 0; k < active.size(); ++k) {
            current_ = active[k];
            values[k] = evaluate(e, *this);
        }
        return values;
    }

    void stmt(const Stmt& s, const std::vector<int64_t>& active) {
        switch (s.kind) {
            case StmtKind::For: {
                if (s.loop == LoopKind::Bound) throw IRError("interpreter: " + kernel_.name + ": nested bound loop");
                const size_t v = id(s.var);
                for (int64_t i = 0; i < s.extent; ++i) {
                    for (int64_t t : active) env_[t * ids_.size() + v] = makeI(ScalarType::I64, i);
                    for (const auto& inner : s.body) stmt(*inner, active);
                }
                break;
            }
            case StmtKind::Let:
            case StmtKind::Assign: setAll(id(s.name), active, evalAll(*s.value, active)); break;
            case StmtKind::Store:
            case StmtKind::AtomicAdd: {
                auto index = evalAll(*s.index, active);
                auto value = evalAll(*s.value, active);
                for (size_t k = 0; k < active.size(); ++k) write(s.buffer, index[k].i, value[k], s.kind == StmtKind::AtomicAdd, active[k]);
                break;
            }
            case StmtKind::If: {
                auto cond = evalAll(*s.cond, active);
                std::vector<int64_t> taken, other;
                for (size_t k = 0; k < active.size(); ++k) (cond[k].i ? taken : other).push_back(active[k]);
                // The active flags of the enclosing set stay on during a branch only for its own threads.
                for (int64_t t : active) active_[t] = 0;
                body(s.body, taken);
                body(s.elseBody, other);
                for (int64_t t : active) active_[t] = 1;
                break;
            }
            case StmtKind::Barrier:
                if (static_cast<int64_t>(active.size()) != threads_) {
                    throw IRError("interpreter: " + kernel_.name + ": barrier reached by " + std::to_string(active.size()) + " of " +
                                  std::to_string(threads_) + " threads (divergent barrier)");
                }
                ++epoch_; // accesses on either side are ordered
                break;
        }
    }

    char* address(int buffer, int64_t index, int64_t thread) {
        const BufferDecl& b = kernel_.buffers[buffer];
        if (index < 0 || index >= b.elements()) {
            throw IRError("interpreter: " + kernel_.name + " accesses " + b.name + "[" + std::to_string(index) + "], outside its " +
                          std::to_string(b.elements()) + " elements (thread " + std::to_string(thread) + ")");
        }
        if (b.scope == MemScope::Register) {
            for (auto& [i, mem] : registers_) {
                if (i == static_cast<size_t>(buffer)) return reinterpret_cast<char*>(mem.data() + thread * b.elements() + index);
            }
        }
        return memory_[buffer] + index * 4;
    }

    void write(int buffer, int64_t index, const Scalar& value, bool accumulate, int64_t thread) {
        char* p = address(buffer, index, thread);
        track(buffer, index, thread, accumulate ? Access::Atomic : Access::Write);
        if (kernel_.buffers[buffer].type == ScalarType::F32) {
            if (accumulate) {
                // Blocks may run on several host threads.
                std::atomic_ref<float>(*reinterpret_cast<float*>(p)).fetch_add(value.f);
                return;
            }
            std::memcpy(p, &value.f, 4);
        } else {
            if (accumulate) throw IRError("interpreter: integer atomics are not supported");
            int32_t v = static_cast<int32_t>(value.i);
            std::memcpy(p, &v, 4);
        }
    }

    // --- race detection ---
    enum class Access { Read, Write, Atomic };
    static constexpr int32_t kNone = -1, kMany = -2;

    // Shared memory: the last write and the readers of each element since the last barrier (epoch).
    struct SharedAccess {
        int64_t writeEpoch{-1};
        int32_t writer{kNone};
        int64_t readEpoch{-1};
        int32_t reader{kNone}; // kMany: several threads
    };
    // Global memory, within the current block: who wrote, read and atomically added to each element.
    struct GlobalAccess {
        int32_t writer{kNone};
        int32_t reader{kNone};
        int32_t adder{kNone};
    };

    std::string threadName(int64_t t) const {
        const auto& bl = kernel_.block;
        const auto& g = kernel_.grid;
        return "thread (" + std::to_string(t % bl[0]) + "," + std::to_string((t / bl[0]) % bl[1]) + "," +
               std::to_string(t / (bl[0] * bl[1])) + ") of block (" + std::to_string(block_ % g[0]) + "," +
               std::to_string((block_ / g[0]) % g[1]) + "," + std::to_string(block_ / (g[0] * g[1])) + ")";
    }

    [[noreturn]] void race(const std::string& what, int buffer, int64_t index, int64_t thread, int32_t other) const {
        std::string others = other == kMany ? "other threads" : threadName(other);
        throw IRError("race: " + kernel_.name + ": " + threadName(thread) + " " + what + " " + kernel_.buffers[buffer].name +
                      "[" + std::to_string(index) + "]; " + others + " access it too, unordered");
    }

    void track(int buffer, int64_t index, int64_t thread, Access kind) {
        const BufferDecl& b = kernel_.buffers[buffer];
        const auto t = static_cast<int32_t>(thread);
        auto other = [t](int32_t who) { return who != kNone && who != t; };
        if (b.scope == MemScope::Register) return; // private to the thread
        if (b.scope == MemScope::Shared) {
            SharedAccess& a = sharedAccess_[sharedSlot_[buffer]][index];
            if (kind == Access::Read) {
                if (a.writeEpoch == epoch_ && other(a.writer)) race("reads (no barrier since the write)", buffer, index, thread, a.writer);
                if (a.readEpoch != epoch_) {
                    a.readEpoch = epoch_;
                    a.reader = t;
                } else if (a.reader != t) {
                    a.reader = kMany;
                }
            } else {
                if (a.writeEpoch == epoch_ && other(a.writer)) race("writes (no barrier since another write)", buffer, index, thread, a.writer);
                if (a.readEpoch == epoch_ && other(a.reader)) race("writes (no barrier since a read)", buffer, index, thread, a.reader);
                a.writeEpoch = epoch_;
                a.writer = t;
            }
            return;
        }
        GlobalAccess& a = globalAccess_[(static_cast<uint64_t>(buffer) << 40) | static_cast<uint64_t>(index)];
        switch (kind) {
            case Access::Read:
                if (other(a.writer)) race("reads", buffer, index, thread, a.writer);
                if (other(a.adder)) race("reads", buffer, index, thread, a.adder);
                a.reader = a.reader == kNone || a.reader == t ? t : kMany;
                break;
            case Access::Write:
                if (other(a.writer)) race("writes", buffer, index, thread, a.writer);
                if (other(a.reader)) race("writes", buffer, index, thread, a.reader);
                if (other(a.adder)) race("writes", buffer, index, thread, a.adder);
                a.writer = t;
                break;
            case Access::Atomic:
                if (other(a.writer)) race("atomically adds to", buffer, index, thread, a.writer);
                if (other(a.reader)) race("atomically adds to", buffer, index, thread, a.reader);
                a.adder = a.adder == kNone || a.adder == t ? t : kMany;
                break;
        }
    }

    const Kernel& kernel_;
    GpuLaunchCheck* launch_{nullptr};
    std::unique_ptr<GpuLaunchCheck> ownLaunch_;
    std::vector<int> sharedSlot_;
    std::vector<std::vector<SharedAccess>> sharedAccess_;
    std::unordered_map<uint64_t, GlobalAccess> globalAccess_;
    int64_t epoch_{0};
    int64_t block_{0};
    std::vector<char*> memory_;
    std::vector<std::pair<size_t, std::vector<float>>> shared_;
    std::vector<std::pair<size_t, std::vector<float>>> registers_;
    std::unordered_map<std::string, size_t> ids_;
    std::array<int64_t, 6> boundVars_{-1, -1, -1, -1, -1, -1};
    const Body* threadBody_{nullptr};
    int64_t threads_{1};
    std::vector<Scalar> env_;
    std::vector<char> active_;
    int64_t current_{0};
};

// --- The warp probe ----------------------------------------------------------------------------------------------
//
// Runs each warp of block 0 on its own, in lockstep, with symbolic memory: loads read 0, stores are dropped, every
// loop runs at most `iterations` times, barriers are ignored. Each execution of a load or store by the warp is one
// request; the indices of its active lanes are what bank-conflict and coalescing analysis need. Index arithmetic of
// the schedules depends on thread and loop indices, not on data, so a couple of iterations show the pattern.
class WarpProbe {
public:
    WarpProbe(const Kernel& kernel, int64_t iterations) : kernel_(kernel), iterations_(iterations) {
        const Body* b = &kernel.body;
        while (b->size() == 1 && (*b)[0]->kind == StmtKind::For && (*b)[0]->loop == LoopKind::Bound) {
            bound_.emplace_back((*b)[0]->var, static_cast<int>((*b)[0]->axis));
            b = &(*b)[0]->body;
        }
        body_ = b;
    }

    std::vector<WarpRequest> run() {
        const auto& bl = kernel_.block;
        const int64_t threads = bl[0] * bl[1] * bl[2];
        for (int64_t w = 0; w * 32 < threads; ++w) {
            warp_ = static_cast<int>(w);
            const int64_t n = std::min<int64_t>(32, threads - w * 32);
            env_.assign(n, {});
            std::vector<int> lanes;
            for (int64_t lane = 0; lane < n; ++lane) {
                const int64_t t = w * 32 + lane;
                const int64_t tc[3] = {t % bl[0], (t / bl[0]) % bl[1], t / (bl[0] * bl[1])};
                for (const auto& [name, axis] : bound_) env_[lane][name] = makeI(ScalarType::I64, axis < 3 ? 0 : tc[axis - 3]);
                lanes.push_back(static_cast<int>(lane));
            }
            body(*body_, lanes);
        }
        return std::move(requests_);
    }

    Scalar variable(const Expr& e) {
        auto it = env_[lane_].find(e.name);
        if (it == env_[lane_].end()) throw IRError("warp probe: " + kernel_.name + ": undefined variable '" + e.name + "'");
        return it->second;
    }

    Scalar load(const Expr& e, int64_t index) {
        record(&e, e.buffer, false, index);
        return e.type == ScalarType::F32 ? makeF(0.0f) : makeI(e.type, 0);
    }

    Scalar shuffle(const Expr& e) { return evaluate(*e.args[0], *this); } // the lane's own value: indices don't depend on it

private:
    void record(const void* site, int buffer, bool isStore, int64_t index) {
        if (kernel_.buffers[buffer].scope == MemScope::Register) return;
        auto [it, inserted] = pending_.try_emplace(site);
        WarpRequest& r = it->second;
        if (inserted) {
            r.site = site;
            r.buffer = buffer;
            r.store = isStore;
            r.warp = warp_;
            order_.push_back(site);
        }
        if (std::find(r.lanes.begin(), r.lanes.end(), lane_) != r.lanes.end()) return; // one access per lane and request
        r.lanes.push_back(lane_);
        r.indices.push_back(index);
    }

    // The requests of one statement, in the order their sites were first seen.
    void flush() {
        for (const void* site : order_) requests_.push_back(std::move(pending_[site]));
        pending_.clear();
        order_.clear();
    }

    std::vector<Scalar> evalAll(const Expr& e, const std::vector<int>& lanes) {
        std::vector<Scalar> out;
        for (int lane : lanes) {
            lane_ = lane;
            out.push_back(evaluate(e, *this));
        }
        return out;
    }

    void body(const Body& stmts, const std::vector<int>& lanes) {
        if (lanes.empty()) return;
        for (const auto& s : stmts) stmt(*s, lanes);
    }

    void stmt(const Stmt& s, const std::vector<int>& lanes) {
        switch (s.kind) {
            case StmtKind::For: {
                if (s.loop == LoopKind::Bound) throw IRError("warp probe: " + kernel_.name + ": nested bound loop");
                for (int64_t i = 0; i < std::min(s.extent, iterations_); ++i) {
                    for (int lane : lanes) env_[lane][s.var] = makeI(ScalarType::I64, i);
                    body(s.body, lanes);
                }
                break;
            }
            case StmtKind::Let:
            case StmtKind::Assign: {
                auto values = evalAll(*s.value, lanes);
                for (size_t k = 0; k < lanes.size(); ++k) env_[lanes[k]][s.name] = values[k];
                flush();
                break;
            }
            case StmtKind::Store:
            case StmtKind::AtomicAdd: {
                auto index = evalAll(*s.index, lanes);
                evalAll(*s.value, lanes);
                for (size_t k = 0; k < lanes.size(); ++k) {
                    lane_ = lanes[k];
                    record(&s, s.buffer, true, index[k].i);
                }
                flush();
                break;
            }
            case StmtKind::If: {
                auto cond = evalAll(*s.cond, lanes);
                flush();
                std::vector<int> taken, other;
                for (size_t k = 0; k < lanes.size(); ++k) (cond[k].i ? taken : other).push_back(lanes[k]);
                body(s.body, taken);
                body(s.elseBody, other);
                break;
            }
            case StmtKind::Barrier: break;
        }
    }

    const Kernel& kernel_;
    int64_t iterations_;
    std::vector<std::pair<std::string, int>> bound_;
    const Body* body_{nullptr};
    std::vector<std::unordered_map<std::string, Scalar>> env_;
    int lane_{0};
    int warp_{0};
    std::unordered_map<const void*, WarpRequest> pending_;
    std::vector<const void*> order_;
    std::vector<WarpRequest> requests_;
};

} // namespace

GpuLaunchCheck::GpuLaunchCheck(const Kernel& kernel) : kernel_(kernel) {
    for (const BufferDecl& b : kernel.buffers) flags_.emplace_back(b.scope == MemScope::Global ? b.elements() : 0);
}

void GpuLaunchCheck::merge(int buffer, int64_t index, uint8_t flags, int64_t block) {
    // Every block merges each element once, so what is already recorded comes from other blocks.
    uint8_t before = flags_[buffer][index].fetch_or(flags, std::memory_order_relaxed);
    bool conflict = ((before & kWrite) && flags) || ((flags & kWrite) && before) || ((before & kAtomic) && (flags & kRead)) ||
                    ((flags & kAtomic) && (before & kRead));
    if (!conflict) return;
    auto kinds = [](uint8_t f) {
        std::string out;
        for (auto [bit, name] : {std::pair{kRead, "reads"}, {kWrite, "writes"}, {kAtomic, "atomically adds to"}}) {
            if (f & bit) out += (out.empty() ? "" : " and ") + std::string(name);
        }
        return out;
    };
    const auto& g = kernel_.grid;
    throw IRError("race: " + kernel_.name + ": block (" + std::to_string(block % g[0]) + "," + std::to_string((block / g[0]) % g[1]) +
                  "," + std::to_string(block / (g[0] * g[1])) + ") " + kinds(flags) + " " + kernel_.buffers[buffer].name + "[" +
                  std::to_string(index) + "], which another block of the launch " + kinds(before) + " (blocks are unordered)");
}

std::vector<WarpRequest> probeWarpAccesses(const Kernel& kernel, int64_t iterations) {
    if (!kernel.gpu) throw IRError("warp probe: " + kernel.name + " is not a GPU kernel");
    if (kernel.body.empty()) return {};
    return WarpProbe(kernel, iterations).run();
}

void interpret(const Kernel& kernel, const std::vector<void*>& args) {
    interpret(kernel, args, 0, std::numeric_limits<int64_t>::max());
}

void interpret(const Kernel& kernel, const std::vector<void*>& args, int64_t begin, int64_t end, GpuLaunchCheck* launch) {
    if (kernel.gpu) GpuInterpreter(kernel, args, launch).run(begin, end);
    else Interpreter(kernel, args).run(begin, end);
}

} // namespace dlc::loopir
