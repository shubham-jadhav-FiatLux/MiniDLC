#pragma once

#include "dlc/ir/Type.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

// Loop IR v2: a small loop-nest IR (a "TIR-lite") for both targets. Schedules turn each fused kernel's computation
// into a nest of loops whose kinds say how they run (serially, across CPU threads, vectorized, unrolled, or bound to
// GPU blocks and threads); the emitters translate it mechanically.
namespace dlc::loopir {

// Index arithmetic is I64; tensor elements are F32 or I32; conditions are Bool.
enum class ScalarType { F32, I32, I64, Bool };

std::string scalarTypeName(ScalarType type);
ScalarType elementType(DataType type);

enum class ExprKind { Int, Float, Var, Load, Binary, Unary, Select, Compare, Cast, Shuffle };
enum class BinaryOp { Add, Sub, Mul, Div, Rem, Min, Max, MaxNaN, Fma };
enum class UnaryOp { Neg, Exp, Log, Tanh, Sqrt, Rsqrt };
enum class CompareOp { LT, LE, GT, GE, EQ, NE };

struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;

// An immutable expression tree. Load reads buffer `buffer` at the flat element index args[0]; Fma is a*b+c with
// args {a, b, c}; MaxNaN is max(a, b) that returns NaN if either is NaN. Shuffle (GPU) reads args[0] from another
// lane of the warp: `args[1]` lanes down (keeping its own value past the end), or the lane whose index is this
// lane's XOR args[1] (a butterfly step, after which every lane holds the reduction).
struct Expr {
    ExprKind kind{ExprKind::Int};
    ScalarType type{ScalarType::I64};
    int64_t intValue{0};
    double floatValue{0.0};
    std::string name; // Var
    int buffer{-1};   // Load
    BinaryOp binary{BinaryOp::Add};
    UnaryOp unary{UnaryOp::Neg};
    CompareOp compare{CompareOp::LT};
    bool butterfly{false}; // Shuffle: XOR lane instead of lane + delta
    std::vector<ExprPtr> args;
};

// Constructors fold integer constants and drop identities (x+0, x*1, x*0 for indices).
ExprPtr intConst(int64_t value);
ExprPtr floatConst(double value);
ExprPtr var(const std::string& name, ScalarType type = ScalarType::I64);
ExprPtr load(int buffer, ExprPtr index, ScalarType type);
ExprPtr binary(BinaryOp op, ExprPtr a, ExprPtr b);
ExprPtr fma(ExprPtr a, ExprPtr b, ExprPtr c);
ExprPtr unary(UnaryOp op, ExprPtr a);
ExprPtr select(ExprPtr cond, ExprPtr a, ExprPtr b);
ExprPtr compare(CompareOp op, ExprPtr a, ExprPtr b);
ExprPtr cast(ScalarType type, ExprPtr a);
ExprPtr shuffleDown(ExprPtr value, ExprPtr delta);
ExprPtr shuffleXor(ExprPtr value, ExprPtr laneMask);
inline ExprPtr add(ExprPtr a, ExprPtr b) { return binary(BinaryOp::Add, std::move(a), std::move(b)); }
inline ExprPtr sub(ExprPtr a, ExprPtr b) { return binary(BinaryOp::Sub, std::move(a), std::move(b)); }
inline ExprPtr mul(ExprPtr a, ExprPtr b) { return binary(BinaryOp::Mul, std::move(a), std::move(b)); }
inline ExprPtr div(ExprPtr a, ExprPtr b) { return binary(BinaryOp::Div, std::move(a), std::move(b)); }
inline ExprPtr rem(ExprPtr a, ExprPtr b) { return binary(BinaryOp::Rem, std::move(a), std::move(b)); }

enum class LoopKind { Serial, Parallel, Vectorized, Unrolled, Bound };
enum class BindAxis { BlockX, BlockY, BlockZ, ThreadX, ThreadY, ThreadZ };

std::string bindAxisName(BindAxis axis);

enum class StmtKind { For, Let, Assign, Store, AtomicAdd, If, Barrier };

struct Stmt;
using StmtPtr = std::shared_ptr<Stmt>;
using Body = std::vector<StmtPtr>;

// For: `var` runs over [0, extent) (normalized loops; schedules express tiles as outer*factor+inner). A Vectorized
// loop has a vector width, an Unrolled loop is fully unrolled, a Bound loop is one thread or block index.
// Let declares a mutable local scalar `name`; Assign updates it. Store/AtomicAdd write `value` at `buffer[index]`.
struct Stmt {
    StmtKind kind{StmtKind::For};
    // For
    std::string var;
    int64_t extent{0};
    LoopKind loop{LoopKind::Serial};
    int width{1};
    BindAxis axis{BindAxis::ThreadX};
    Body body;
    // Let / Assign
    std::string name;
    ScalarType type{ScalarType::F32};
    // Let / Assign / Store / AtomicAdd
    ExprPtr value;
    // Store / AtomicAdd
    int buffer{-1};
    ExprPtr index;
    // If
    ExprPtr cond;
    Body elseBody;
};

StmtPtr forLoop(const std::string& var, int64_t extent, LoopKind kind, Body body, int width = 1);
StmtPtr boundLoop(const std::string& var, int64_t extent, BindAxis axis, Body body);
StmtPtr let(const std::string& name, ScalarType type, ExprPtr value);
StmtPtr assign(const std::string& name, ExprPtr value);
StmtPtr store(int buffer, ExprPtr index, ExprPtr value);
StmtPtr atomicAdd(int buffer, ExprPtr index, ExprPtr value);
StmtPtr ifThen(ExprPtr cond, Body thenBody, Body elseBody = {});
StmtPtr barrier();

enum class MemScope { Global, Shared, Register };

std::string memScopeName(MemScope scope);

struct BufferDecl {
    std::string name;
    ScalarType type{ScalarType::F32};
    Shape shape;
    MemScope scope{MemScope::Global};

    int64_t elements() const;
};

// One parallel region: the global buffers are its parameters, in order; shared and register buffers are local. On the
// CPU the outermost loop may be Parallel, and the runtime splits its range across threads; on the GPU the Bound loops
// give the launch configuration.
struct Kernel {
    std::string name;
    std::vector<BufferDecl> buffers;
    Body body;
    bool gpu{false}; // bound loops: grid and block below
    std::array<int64_t, 3> grid{1, 1, 1};
    std::array<int64_t, 3> block{1, 1, 1};
    double flops{0.0};
    int64_t bytes{0};

    int addBuffer(BufferDecl decl);
    std::vector<int> globalBuffers() const;
    // The extent of the outermost loop if it is Parallel, else 1.
    int64_t parallelExtent() const;
};

std::string toString(const ExprPtr& expr, const Kernel& kernel);
std::string toString(const Kernel& kernel);

} // namespace dlc::loopir
