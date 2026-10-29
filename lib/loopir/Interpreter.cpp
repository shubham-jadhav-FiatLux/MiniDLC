#include "dlc/loopir/Interpreter.h"
#include "dlc/ir/Error.h"
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

    Scalar eval(const Expr& e) {
        switch (e.kind) {
            case ExprKind::Int: return makeI(ScalarType::I64, e.intValue);
            case ExprKind::Float: return makeF(static_cast<float>(e.floatValue));
            case ExprKind::Var: return lookup(e.name);
            case ExprKind::Load: return read(e.buffer, eval(*e.args[0]).i);
            case ExprKind::Binary: return binary(e);
            case ExprKind::Unary: {
                Scalar a = eval(*e.args[0]);
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
            case ExprKind::Select: return eval(*e.args[0]).i ? eval(*e.args[1]) : eval(*e.args[2]);
            case ExprKind::Compare: {
                Scalar a = eval(*e.args[0]);
                Scalar b = eval(*e.args[1]);
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
                Scalar a = eval(*e.args[0]);
                if (e.type == ScalarType::F32) return makeF(a.type == ScalarType::F32 ? a.f : static_cast<float>(a.i));
                if (a.type == ScalarType::F32) return makeI(e.type, static_cast<int64_t>(std::trunc(a.f)));
                return makeI(e.type, e.type == ScalarType::I32 ? static_cast<int32_t>(a.i) : a.i);
            }
            case ExprKind::Shuffle: throw IRError("interpreter: warp shuffles need the GPU interpreter (P5)");
        }
        throw IRError("interpreter: unknown expression");
    }

    Scalar binary(const Expr& e) {
        Scalar a = eval(*e.args[0]);
        Scalar b = eval(*e.args[1]);
        if (e.binary == BinaryOp::Fma) return makeF(std::fma(a.f, b.f, eval(*e.args[2]).f));
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

    const Kernel& kernel_;
    std::vector<char*> memory_;
    std::vector<std::vector<char>> locals_;
    std::vector<std::unordered_map<std::string, Scalar>> scopes_;
};

} // namespace

void interpret(const Kernel& kernel, const std::vector<void*>& args) {
    Interpreter(kernel, args).run(0, std::numeric_limits<int64_t>::max());
}

void interpret(const Kernel& kernel, const std::vector<void*>& args, int64_t begin, int64_t end) {
    Interpreter(kernel, args).run(begin, end);
}

} // namespace dlc::loopir
