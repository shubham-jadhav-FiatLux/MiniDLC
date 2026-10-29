#include "dlc/loopir/LoopIR.h"
#include "dlc/ir/Error.h"
#include <cmath>
#include <sstream>

namespace dlc::loopir {

std::string scalarTypeName(ScalarType type) {
    switch (type) {
        case ScalarType::F32: return "f32";
        case ScalarType::I32: return "i32";
        case ScalarType::I64: return "i64";
        case ScalarType::Bool: return "bool";
    }
    return "?";
}

ScalarType elementType(DataType type) { return type == DataType::Int32 ? ScalarType::I32 : ScalarType::F32; }

std::string bindAxisName(BindAxis axis) {
    switch (axis) {
        case BindAxis::BlockX: return "blockIdx.x";
        case BindAxis::BlockY: return "blockIdx.y";
        case BindAxis::BlockZ: return "blockIdx.z";
        case BindAxis::ThreadX: return "threadIdx.x";
        case BindAxis::ThreadY: return "threadIdx.y";
        case BindAxis::ThreadZ: return "threadIdx.z";
    }
    return "?";
}

std::string memScopeName(MemScope scope) {
    switch (scope) {
        case MemScope::Global: return "global";
        case MemScope::Shared: return "shared";
        case MemScope::Register: return "register";
    }
    return "?";
}

namespace {

std::shared_ptr<Expr> make(ExprKind kind, ScalarType type) {
    auto e = std::make_shared<Expr>();
    e->kind = kind;
    e->type = type;
    return e;
}

bool isInt(const ExprPtr& e, int64_t v) { return e->kind == ExprKind::Int && e->intValue == v; }

} // namespace

ExprPtr intConst(int64_t value) {
    auto e = make(ExprKind::Int, ScalarType::I64);
    e->intValue = value;
    return e;
}

ExprPtr floatConst(double value) {
    auto e = make(ExprKind::Float, ScalarType::F32);
    e->floatValue = value;
    return e;
}

ExprPtr var(const std::string& name, ScalarType type) {
    auto e = make(ExprKind::Var, type);
    e->name = name;
    return e;
}

ExprPtr load(int buffer, ExprPtr index, ScalarType type) {
    auto e = make(ExprKind::Load, type);
    e->buffer = buffer;
    e->args = {std::move(index)};
    return e;
}

ExprPtr binary(BinaryOp op, ExprPtr a, ExprPtr b) {
    if (a->type != b->type) throw IRError("loop IR: binary operands of different types");
    if (a->type == ScalarType::I64) {
        if (a->kind == ExprKind::Int && b->kind == ExprKind::Int) {
            int64_t x = a->intValue;
            int64_t y = b->intValue;
            switch (op) {
                case BinaryOp::Add: return intConst(x + y);
                case BinaryOp::Sub: return intConst(x - y);
                case BinaryOp::Mul: return intConst(x * y);
                case BinaryOp::Div: if (y != 0) return intConst(x / y); break;
                case BinaryOp::Rem: if (y != 0) return intConst(x % y); break;
                case BinaryOp::Min: return intConst(std::min(x, y));
                case BinaryOp::Max: return intConst(std::max(x, y));
                default: break;
            }
        }
        if (op == BinaryOp::Add && isInt(a, 0)) return b;
        if ((op == BinaryOp::Add || op == BinaryOp::Sub) && isInt(b, 0)) return a;
        if (op == BinaryOp::Mul && (isInt(a, 0) || isInt(b, 0))) return intConst(0);
        if (op == BinaryOp::Mul && isInt(a, 1)) return b;
        if ((op == BinaryOp::Mul || op == BinaryOp::Div) && isInt(b, 1)) return a;
        if (op == BinaryOp::Rem && isInt(b, 1)) return intConst(0);
    }
    auto e = make(ExprKind::Binary, a->type);
    e->binary = op;
    e->args = {std::move(a), std::move(b)};
    return e;
}

ExprPtr fma(ExprPtr a, ExprPtr b, ExprPtr c) {
    auto e = make(ExprKind::Binary, ScalarType::F32);
    e->binary = BinaryOp::Fma;
    e->args = {std::move(a), std::move(b), std::move(c)};
    return e;
}

ExprPtr unary(UnaryOp op, ExprPtr a) {
    auto e = make(ExprKind::Unary, a->type);
    e->unary = op;
    e->args = {std::move(a)};
    return e;
}

ExprPtr select(ExprPtr cond, ExprPtr a, ExprPtr b) {
    if (cond->type != ScalarType::Bool) throw IRError("loop IR: select needs a bool condition");
    auto e = make(ExprKind::Select, a->type);
    e->args = {std::move(cond), std::move(a), std::move(b)};
    return e;
}

ExprPtr compare(CompareOp op, ExprPtr a, ExprPtr b) {
    if (a->type != b->type) throw IRError("loop IR: compare operands of different types");
    auto e = make(ExprKind::Compare, ScalarType::Bool);
    e->compare = op;
    e->args = {std::move(a), std::move(b)};
    return e;
}

ExprPtr cast(ScalarType type, ExprPtr a) {
    if (a->type == type) return a;
    if (a->kind == ExprKind::Int && type == ScalarType::F32) return floatConst(static_cast<double>(a->intValue));
    auto e = make(ExprKind::Cast, type);
    e->args = {std::move(a)};
    return e;
}

ExprPtr shuffleDown(ExprPtr value, ExprPtr delta) {
    auto e = make(ExprKind::Shuffle, value->type);
    e->args = {std::move(value), std::move(delta)};
    return e;
}

StmtPtr forLoop(const std::string& v, int64_t extent, LoopKind kind, Body body, int width) {
    auto s = std::make_shared<Stmt>();
    s->kind = StmtKind::For;
    s->var = v;
    s->extent = extent;
    s->loop = kind;
    s->width = width;
    s->body = std::move(body);
    return s;
}

StmtPtr boundLoop(const std::string& v, int64_t extent, BindAxis axis, Body body) {
    auto s = forLoop(v, extent, LoopKind::Bound, std::move(body));
    s->axis = axis;
    return s;
}

StmtPtr let(const std::string& name, ScalarType type, ExprPtr value) {
    auto s = std::make_shared<Stmt>();
    s->kind = StmtKind::Let;
    s->name = name;
    s->type = type;
    s->value = std::move(value);
    return s;
}

StmtPtr assign(const std::string& name, ExprPtr value) {
    auto s = std::make_shared<Stmt>();
    s->kind = StmtKind::Assign;
    s->name = name;
    s->type = value->type;
    s->value = std::move(value);
    return s;
}

StmtPtr store(int buffer, ExprPtr index, ExprPtr value) {
    auto s = std::make_shared<Stmt>();
    s->kind = StmtKind::Store;
    s->buffer = buffer;
    s->index = std::move(index);
    s->value = std::move(value);
    return s;
}

StmtPtr atomicAdd(int buffer, ExprPtr index, ExprPtr value) {
    auto s = store(buffer, std::move(index), std::move(value));
    s->kind = StmtKind::AtomicAdd;
    return s;
}

StmtPtr ifThen(ExprPtr cond, Body thenBody, Body elseBody) {
    auto s = std::make_shared<Stmt>();
    s->kind = StmtKind::If;
    s->cond = std::move(cond);
    s->body = std::move(thenBody);
    s->elseBody = std::move(elseBody);
    return s;
}

StmtPtr barrier() {
    auto s = std::make_shared<Stmt>();
    s->kind = StmtKind::Barrier;
    return s;
}

int64_t BufferDecl::elements() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

int Kernel::addBuffer(BufferDecl decl) {
    buffers.push_back(std::move(decl));
    return static_cast<int>(buffers.size()) - 1;
}

std::vector<int> Kernel::globalBuffers() const {
    std::vector<int> out;
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i].scope == MemScope::Global) out.push_back(static_cast<int>(i));
    }
    return out;
}

int64_t Kernel::parallelExtent() const {
    if (body.size() == 1 && body[0]->kind == StmtKind::For && body[0]->loop == LoopKind::Parallel) {
        return body[0]->extent;
    }
    return 1;
}

// --- Printing ----------------------------------------------------------------------------------------------------

namespace {

std::string binaryName(BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return "+";
        case BinaryOp::Sub: return "-";
        case BinaryOp::Mul: return "*";
        case BinaryOp::Div: return "/";
        case BinaryOp::Rem: return "%";
        case BinaryOp::Min: return "min";
        case BinaryOp::Max: return "max";
        case BinaryOp::MaxNaN: return "max_nan";
        case BinaryOp::Fma: return "fma";
    }
    return "?";
}

std::string unaryName(UnaryOp op) {
    switch (op) {
        case UnaryOp::Neg: return "neg";
        case UnaryOp::Exp: return "exp";
        case UnaryOp::Log: return "log";
        case UnaryOp::Tanh: return "tanh";
        case UnaryOp::Sqrt: return "sqrt";
        case UnaryOp::Rsqrt: return "rsqrt";
    }
    return "?";
}

std::string compareName(CompareOp op) {
    switch (op) {
        case CompareOp::LT: return "<";
        case CompareOp::LE: return "<=";
        case CompareOp::GT: return ">";
        case CompareOp::GE: return ">=";
        case CompareOp::EQ: return "==";
        case CompareOp::NE: return "!=";
    }
    return "?";
}

std::string formatFloat(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::ostringstream os;
    os.precision(9);
    os << v;
    std::string s = os.str();
    if (s.find_first_of(".en") == std::string::npos) s += ".0";
    return s;
}

class Printer {
public:
    explicit Printer(const Kernel& kernel) : kernel_(kernel) {}

    std::string expr(const ExprPtr& e) const {
        switch (e->kind) {
            case ExprKind::Int: return std::to_string(e->intValue);
            case ExprKind::Float: return formatFloat(e->floatValue);
            case ExprKind::Var: return e->name;
            case ExprKind::Load: return kernel_.buffers[e->buffer].name + "[" + expr(e->args[0]) + "]";
            case ExprKind::Binary: {
                if (e->binary == BinaryOp::Fma) {
                    return "fma(" + expr(e->args[0]) + ", " + expr(e->args[1]) + ", " + expr(e->args[2]) + ")";
                }
                if (e->binary == BinaryOp::Min || e->binary == BinaryOp::Max || e->binary == BinaryOp::MaxNaN) {
                    return binaryName(e->binary) + "(" + expr(e->args[0]) + ", " + expr(e->args[1]) + ")";
                }
                return "(" + expr(e->args[0]) + " " + binaryName(e->binary) + " " + expr(e->args[1]) + ")";
            }
            case ExprKind::Unary: return unaryName(e->unary) + "(" + expr(e->args[0]) + ")";
            case ExprKind::Select:
                return "select(" + expr(e->args[0]) + ", " + expr(e->args[1]) + ", " + expr(e->args[2]) + ")";
            case ExprKind::Compare:
                return "(" + expr(e->args[0]) + " " + compareName(e->compare) + " " + expr(e->args[1]) + ")";
            case ExprKind::Cast: return scalarTypeName(e->type) + "(" + expr(e->args[0]) + ")";
            case ExprKind::Shuffle: return "shfl_down(" + expr(e->args[0]) + ", " + expr(e->args[1]) + ")";
        }
        return "?";
    }

    void body(std::ostream& os, const Body& stmts, int depth) const {
        for (const auto& s : stmts) stmt(os, *s, depth);
    }

    void stmt(std::ostream& os, const Stmt& s, int depth) const {
        std::string pad(2 * depth, ' ');
        switch (s.kind) {
            case StmtKind::For: {
                os << pad;
                switch (s.loop) {
                    case LoopKind::Serial: os << "for "; break;
                    case LoopKind::Parallel: os << "parallel for "; break;
                    case LoopKind::Vectorized: os << "for "; break;
                    case LoopKind::Unrolled: os << "for "; break;
                    case LoopKind::Bound: os << "for "; break;
                }
                os << s.var << " in 0.." << s.extent;
                if (s.loop == LoopKind::Vectorized) os << " vectorize(" << s.width << ")";
                if (s.loop == LoopKind::Unrolled) os << " unroll";
                if (s.loop == LoopKind::Bound) os << " bind(" << bindAxisName(s.axis) << ")";
                os << " {\n";
                body(os, s.body, depth + 1);
                os << pad << "}\n";
                break;
            }
            case StmtKind::Let:
                os << pad << scalarTypeName(s.type) << " " << s.name << " = " << expr(s.value) << ";\n";
                break;
            case StmtKind::Assign: os << pad << s.name << " = " << expr(s.value) << ";\n"; break;
            case StmtKind::Store:
                os << pad << kernel_.buffers[s.buffer].name << "[" << expr(s.index) << "] = " << expr(s.value) << ";\n";
                break;
            case StmtKind::AtomicAdd:
                os << pad << "atomic_add(" << kernel_.buffers[s.buffer].name << "[" << expr(s.index) << "], "
                   << expr(s.value) << ");\n";
                break;
            case StmtKind::If:
                os << pad << "if " << expr(s.cond) << " {\n";
                body(os, s.body, depth + 1);
                if (!s.elseBody.empty()) {
                    os << pad << "} else {\n";
                    body(os, s.elseBody, depth + 1);
                }
                os << pad << "}\n";
                break;
            case StmtKind::Barrier: os << pad << "barrier;\n"; break;
        }
    }

private:
    const Kernel& kernel_;
};

std::string shapeText(const Shape& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? "," : "") + std::to_string(shape[i]);
    return s + "]";
}

} // namespace

std::string toString(const ExprPtr& expr, const Kernel& kernel) { return Printer(kernel).expr(expr); }

std::string toString(const Kernel& kernel) {
    std::ostringstream os;
    os << "kernel @" << kernel.name << "(";
    bool first = true;
    for (const BufferDecl& b : kernel.buffers) {
        if (b.scope != MemScope::Global) continue;
        os << (first ? "" : ", ") << b.name << ": " << scalarTypeName(b.type) << shapeText(b.shape);
        first = false;
    }
    os << ")";
    if (kernel.grid != std::array<int64_t, 3>{1, 1, 1} || kernel.block != std::array<int64_t, 3>{1, 1, 1}) {
        os << " grid(" << kernel.grid[0] << "," << kernel.grid[1] << "," << kernel.grid[2] << ") block("
           << kernel.block[0] << "," << kernel.block[1] << "," << kernel.block[2] << ")";
    }
    os << " {\n";
    for (const BufferDecl& b : kernel.buffers) {
        if (b.scope == MemScope::Global) continue;
        os << "  alloc " << b.name << ": " << scalarTypeName(b.type) << shapeText(b.shape) << " "
           << memScopeName(b.scope) << "\n";
    }
    Printer(kernel).body(os, kernel.body, 1);
    os << "}\n";
    return os.str();
}

} // namespace dlc::loopir
