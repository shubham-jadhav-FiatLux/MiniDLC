#include "dlc/backends/llvm/CpuJit.h"
#include "dlc/compiler/Compiler.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/Parser.h"
#include "dlc/loopir/Analysis.h"
#include "dlc/loopir/Interpreter.h"
#include "dlc/loopir/LoopIR.h"
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

using namespace dlc;
using namespace dlc::loopir;

ExprPtr i64(const char* name) { return var(name, ScalarType::I64); }

// out[i] = a[i] + b[i] over n elements, vectorized.
loopir::Kernel vectorAdd(int64_t n) {
    loopir::Kernel k;
    k.name = "vadd";
    int a = k.addBuffer({"%a", ScalarType::F32, {n}, MemScope::Global});
    int b = k.addBuffer({"%b", ScalarType::F32, {n}, MemScope::Global});
    int out = k.addBuffer({"%out", ScalarType::F32, {n}, MemScope::Global});
    k.body = {forLoop("i", n, LoopKind::Vectorized,
                      {store(out, i64("i"), add(load(a, i64("i"), ScalarType::F32), load(b, i64("i"), ScalarType::F32)))},
                      8)};
    return k;
}

TEST(LoopIR, ConstructorsFoldIndexArithmetic) {
    EXPECT_EQ(add(intConst(2), intConst(3))->intValue, 5);
    EXPECT_EQ(mul(i64("i"), intConst(1))->kind, ExprKind::Var);
    EXPECT_EQ(mul(i64("i"), intConst(0))->kind, ExprKind::Int);
    EXPECT_EQ(add(intConst(0), i64("j"))->kind, ExprKind::Var);
}

TEST(LoopIR, PrintsKernels) {
    std::string text = toString(vectorAdd(37));
    EXPECT_NE(text.find("kernel @vadd(%a: f32[37], %b: f32[37], %out: f32[37]) {"), std::string::npos) << text;
    EXPECT_NE(text.find("for i in 0..37 vectorize(8) {"), std::string::npos) << text;
    EXPECT_NE(text.find("%out[i] = (%a[i] + %b[i]);"), std::string::npos) << text;
}

TEST(RangeAnalysis, IntervalArithmetic) {
    RangeEnv env;
    env.set("i", {0, 9});
    env.set("j", {0, 15});
    EXPECT_EQ(rangeOf(add(mul(i64("i"), intConst(16)), i64("j")), env).hi, 159);
    Interval d = rangeOf(div(i64("i"), intConst(4)), env);
    EXPECT_EQ(d.lo, 0);
    EXPECT_EQ(d.hi, 2);
    Interval r = rangeOf(rem(i64("j"), intConst(4)), env);
    EXPECT_EQ(r.lo, 0);
    EXPECT_EQ(r.hi, 3);
    EXPECT_FALSE(rangeOf(i64("unknown"), env).known());
}

TEST(RangeAnalysis, AffineForms) {
    auto a = affineForm(add(mul(add(i64("i"), intConst(2)), intConst(8)), i64("j")));
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(a->coeffs.at("i"), 8);
    EXPECT_EQ(a->coeffs.at("j"), 1);
    EXPECT_EQ(a->constant, 16);
    EXPECT_FALSE(affineForm(div(i64("i"), intConst(2))).has_value());
}

TEST(Bounds, ProvesAffineAccessesAndRejectsOutOfBoundsOnes) {
    loopir::Kernel ok = vectorAdd(64);
    BoundsReport report = checkBounds(ok);
    EXPECT_EQ(report.accesses, 3);
    EXPECT_EQ(report.proven, 3);

    loopir::Kernel bad = vectorAdd(64);
    bad.body[0]->extent = 65; // a schedule bug: one iteration too many
    EXPECT_THROW(checkBounds(bad), IRError);
}

TEST(Bounds, GuardsIndicesReadFromMemory) {
    loopir::Kernel k;
    k.name = "gather";
    int table = k.addBuffer({"%t", ScalarType::F32, {10}, MemScope::Global});
    int ids = k.addBuffer({"%ids", ScalarType::I32, {4}, MemScope::Global});
    int out = k.addBuffer({"%out", ScalarType::F32, {4}, MemScope::Global});
    k.body = {forLoop("p", 4, LoopKind::Serial,
                      {store(out, i64("p"), load(table, cast(ScalarType::I64, load(ids, i64("p"), ScalarType::I32)),
                                                 ScalarType::F32))})};
    BoundsReport report = checkBounds(k);
    EXPECT_EQ(report.guarded, 1);
    // An out-of-range id reads 0 instead of memory outside the table.
    std::vector<float> t(10);
    for (int i = 0; i < 10; ++i) t[i] = static_cast<float>(i);
    std::vector<int32_t> idx = {3, -1, 9, 10};
    std::vector<float> o(4, -5.0f);
    interpret(k, {t.data(), idx.data(), o.data()});
    EXPECT_EQ(o, (std::vector<float>{3.0f, 0.0f, 9.0f, 0.0f}));
}

TEST(Bounds, EliminatesGuardsThatRangeAnalysisProves) {
    loopir::Kernel k;
    k.name = "guards";
    int out = k.addBuffer({"%out", ScalarType::F32, {64}, MemScope::Global});
    // for i in 0..64: if (i < 64) out[i] = 1  — always true: removed.
    //                 if (i < 32) out[i] = 2  — undecided: kept.
    //                 if (i >= 64) out[i] = 3 — always false: removed with its body.
    k.body = {forLoop("i", 64, LoopKind::Serial,
                      {ifThen(compare(CompareOp::LT, i64("i"), intConst(64)), {store(out, i64("i"), floatConst(1.0))}),
                       ifThen(compare(CompareOp::LT, i64("i"), intConst(32)), {store(out, i64("i"), floatConst(2.0))}),
                       ifThen(compare(CompareOp::GE, i64("i"), intConst(64)), {store(out, i64("i"), floatConst(3.0))})})};
    BoundsReport report = checkBounds(k);
    EXPECT_EQ(report.eliminated, 2);
    const Body& body = k.body[0]->body;
    ASSERT_EQ(body.size(), 2u);
    EXPECT_EQ(body[0]->kind, StmtKind::Store);
    EXPECT_EQ(body[1]->kind, StmtKind::If);
}

TEST(Bounds, BranchConditionsNarrowRanges) {
    // for p in 0..9: if (p < 8) out[p] = ... — out has 8 elements, and the guard makes the store provable.
    loopir::Kernel k;
    k.name = "tail";
    int out = k.addBuffer({"%out", ScalarType::F32, {8}, MemScope::Global});
    k.body = {forLoop("p", 9, LoopKind::Serial,
                      {ifThen(compare(CompareOp::LT, i64("p"), intConst(8)), {store(out, i64("p"), floatConst(1.0))})})};
    EXPECT_NO_THROW(checkBounds(k));
}

TEST(Schedule, RejectsParallelIterationsWritingTheSameElement) {
    loopir::Kernel k;
    k.name = "race";
    int out = k.addBuffer({"%out", ScalarType::F32, {1}, MemScope::Global});
    k.body = {forLoop("p", 4, LoopKind::Parallel, {store(out, intConst(0), floatConst(1.0))})};
    EXPECT_THROW(verifySchedule(k), IRError);

    loopir::Kernel ok = vectorAdd(64);
    ok.body[0]->loop = LoopKind::Parallel;
    EXPECT_NO_THROW(verifySchedule(ok));
}

TEST(Schedule, RejectsLoopCarriedDependencesInVectorizedLoops) {
    loopir::Kernel k;
    k.name = "prefix";
    int out = k.addBuffer({"%out", ScalarType::F32, {65}, MemScope::Global});
    // out[i + 1] = out[i] + 1 carries a dependence from one iteration to the next.
    k.body = {forLoop("i", 64, LoopKind::Vectorized,
                      {store(out, add(i64("i"), intConst(1)), add(load(out, i64("i"), ScalarType::F32), floatConst(1.0)))},
                      8)};
    EXPECT_THROW(verifySchedule(k), IRError);
}

TEST(Schedule, AcceptsReductionsIntoRegisterBuffers) {
    loopir::Kernel k;
    k.name = "dot";
    int a = k.addBuffer({"%a", ScalarType::F32, {64}, MemScope::Global});
    int out = k.addBuffer({"%out", ScalarType::F32, {1}, MemScope::Global});
    int acc = k.addBuffer({"acc", ScalarType::F32, {1}, MemScope::Register});
    k.body = {store(acc, intConst(0), floatConst(0.0)),
              forLoop("k", 64, LoopKind::Vectorized,
                      {store(acc, intConst(0), add(load(acc, intConst(0), ScalarType::F32), load(a, i64("k"), ScalarType::F32)))},
                      8),
              store(out, intConst(0), load(acc, intConst(0), ScalarType::F32))};
    EXPECT_NO_THROW(verifySchedule(k));
}

TEST(Interpreter, ChecksEveryAccess) {
    loopir::Kernel k = vectorAdd(8);
    std::vector<float> a(8, 1.0f), b(8, 2.0f), out(8);
    interpret(k, {a.data(), b.data(), out.data()});
    EXPECT_EQ(out, std::vector<float>(8, 3.0f));
    k.body[0]->extent = 9;
    EXPECT_THROW(interpret(k, {a.data(), b.data(), out.data()}), IRError);
}

// A one-block GPU kernel of `threads` threads (x) with buffers a (input) and out, both of `threads` elements.
struct GpuKernel {
    loopir::Kernel k;
    int a, out;
};

GpuKernel oneBlock(int64_t threads) {
    GpuKernel g;
    g.k.name = "gpu";
    g.k.gpu = true;
    g.k.block = {threads, 1, 1};
    g.a = g.k.addBuffer({"%a", ScalarType::F32, {threads}, MemScope::Global});
    g.out = g.k.addBuffer({"%out", ScalarType::F32, {threads}, MemScope::Global});
    return g;
}

void launch(GpuKernel& g, Body body, std::vector<float>& a, std::vector<float>& out) {
    g.k.body = {boundLoop("bx", 1, BindAxis::BlockX, {boundLoop("tx", g.k.block[0], BindAxis::ThreadX, std::move(body))})};
    interpret(g.k, {a.data(), out.data()});
}

TEST(GpuInterpreter, ButterflyShufflesReduceEachWarp) {
    GpuKernel g = oneBlock(64);
    Body body = {let("v", ScalarType::F32, load(g.a, i64("tx"), ScalarType::F32))};
    for (int step = 16; step >= 1; step /= 2) {
        body.push_back(assign("v", add(var("v", ScalarType::F32), shuffleXor(var("v", ScalarType::F32), intConst(step)))));
    }
    body.push_back(store(g.out, i64("tx"), var("v", ScalarType::F32)));
    std::vector<float> a(64), out(64);
    for (int i = 0; i < 64; ++i) a[i] = static_cast<float>(i);
    launch(g, body, a, out);
    for (int i = 0; i < 64; ++i) EXPECT_EQ(out[i], i < 32 ? 496.0f : 1520.0f) << i; // 0+..+31, 32+..+63
}

TEST(GpuInterpreter, ShuffleDownKeepsTheLanesOwnValuePastTheEnd) {
    GpuKernel g = oneBlock(32);
    Body body = {store(g.out, i64("tx"), shuffleDown(load(g.a, i64("tx"), ScalarType::F32), intConst(4)))};
    std::vector<float> a(32), out(32);
    for (int i = 0; i < 32; ++i) a[i] = static_cast<float>(i);
    launch(g, body, a, out);
    for (int i = 0; i < 32; ++i) EXPECT_EQ(out[i], i + 4 < 32 ? i + 4 : i) << i;
}

TEST(GpuInterpreter, SharedMemoryAndBarriers) {
    GpuKernel g = oneBlock(48);
    int s = g.k.addBuffer({"tile", ScalarType::F32, {48}, MemScope::Shared});
    Body body = {store(s, i64("tx"), load(g.a, i64("tx"), ScalarType::F32)), barrier(),
                 store(g.out, i64("tx"), load(s, sub(intConst(47), i64("tx")), ScalarType::F32))};
    std::vector<float> a(48), out(48);
    for (int i = 0; i < 48; ++i) a[i] = static_cast<float>(i);
    launch(g, body, a, out);
    for (int i = 0; i < 48; ++i) EXPECT_EQ(out[i], 47.0f - i) << i;
}

TEST(GpuInterpreter, SharedMemoryStartsAsNaN) {
    GpuKernel g = oneBlock(8);
    int s = g.k.addBuffer({"tile", ScalarType::F32, {8}, MemScope::Shared});
    Body body = {store(g.out, i64("tx"), load(s, i64("tx"), ScalarType::F32))};
    std::vector<float> a(8), out(8);
    launch(g, body, a, out);
    for (float v : out) EXPECT_TRUE(std::isnan(v));
}

TEST(GpuInterpreter, RejectsDivergentBarriersAndShufflesOfInactiveLanes) {
    GpuKernel g = oneBlock(32);
    std::vector<float> a(32), out(32);
    Body divergent = {ifThen(compare(CompareOp::LT, i64("tx"), intConst(16)), {barrier()})};
    try {
        launch(g, divergent, a, out);
        FAIL() << "a barrier reached by half the block must be rejected";
    } catch (const IRError& e) {
        EXPECT_NE(std::string(e.what()).find("divergent barrier"), std::string::npos) << e.what();
    }
    Body inactive = {ifThen(compare(CompareOp::LT, i64("tx"), intConst(16)),
                            {store(g.out, i64("tx"), shuffleXor(load(g.a, i64("tx"), ScalarType::F32), intConst(16)))})};
    EXPECT_THROW(launch(g, inactive, a, out), IRError);
}

TEST(GpuSchedules, MatmulGuardsDisappearForDivisibleShapes) {
    auto lowered = [](const char* ir) {
        ModulePtr m = IRParser::parseModule(ir);
        return compiler::lowerForInspection(*m->getFunctions()[0], Target::GPU);
    };
    auto even = lowered("func @f(%a: f32[128,64], %b: f32[64,128]) -> f32[128,128] {\n  %0 = matmul(%a, %b)\n  return %0\n}\n");
    ASSERT_EQ(even.size(), 1u);
    std::string text = toString(even[0].kernel);
    EXPECT_EQ(text.find("if ("), std::string::npos) << text;
    EXPECT_NE(text.find("barrier"), std::string::npos) << text;
    EXPECT_EQ(even[0].kernel.grid, (std::array<int64_t, 3>{2, 2, 1}));
    auto odd = lowered("func @f(%a: f32[100,60], %b: f32[60,70]) -> f32[100,70] {\n  %0 = matmul(%a, %b)\n  return %0\n}\n");
    EXPECT_NE(toString(odd[0].kernel).find("if ("), std::string::npos);
}

TEST(CpuJit, CompiledKernelMatchesTheInterpreter) {
    std::vector<loopir::Kernel> kernels = {vectorAdd(1000)};
    backends::llvm_emit::CpuJit jit(kernels);
    EXPECT_NE(jit.optimizedIR().find("<8 x float>"), std::string::npos) << "the vectorize(8) loop was not vectorized";
    std::vector<float> a(1000), b(1000), viaJit(1000), viaInterp(1000);
    for (int i = 0; i < 1000; ++i) {
        a[i] = 0.5f * static_cast<float>(i);
        b[i] = std::sin(static_cast<float>(i));
    }
    void* args[] = {a.data(), b.data(), viaJit.data()};
    jit.function(0)(args, 0, 1);
    interpret(kernels[0], {a.data(), b.data(), viaInterp.data()});
    EXPECT_EQ(viaJit, viaInterp);
}

TEST(Compiler, TrainsALinearModel) {
    const char* ir = R"(
func @linear(%x: f32[8,4], %y: i32[8], %w: f32[4,3] {param}) -> f32[] {
  %0 = matmul(%x, %w) : f32[8,3]
  %1 = cross_entropy(%0, %y) : f32[]
  return %1
}
)";
    compiler::CompileOptions options;
    options.optimizer = OptimizerConfig{OptimizerConfig::Kind::SGD, 0.5};
    std::vector<float> x(32);
    for (int i = 0; i < 32; ++i) x[i] = static_cast<float>((i * 7) % 5) - 2.0f;
    std::vector<int32_t> y = {0, 1, 2, 0, 1, 2, 0, 1};
    std::vector<float> w(12, 0.0f);
    std::vector<std::vector<float>> losses;
    for (const char* target : {"cpu", "interp"}) {
        options.target = target;
        auto exe = compiler::compile(ir, options);
        exe->setArg("w", w.data(), w.size() * 4);
        std::unordered_map<std::string, runtime::TensorView> inputs = {
            {"x", {DataType::Float32, {8, 4}, x.data(), x.size() * 4}},
            {"y", {DataType::Int32, {8}, y.data(), y.size() * 4}}};
        std::vector<float> trace;
        for (int step = 0; step < 5; ++step) {
            auto out = exe->run(inputs);
            float loss;
            std::memcpy(&loss, out[0].data.data(), 4);
            trace.push_back(loss);
        }
        losses.push_back(trace);
    }
    EXPECT_NEAR(losses[0][0], std::log(3.0f), 1e-6); // zero weights: uniform predictions
    EXPECT_LT(losses[0][4], losses[0][0]);
    for (int s = 0; s < 5; ++s) EXPECT_NEAR(losses[0][s], losses[1][s], 1e-5) << "step " << s;
}

} // namespace
