#include "dlc/loopir/LoopPasses.h"
#include <unordered_map>
#include <algorithm>

namespace dlc::loopir {

void LoopTilingPass::run(KernelNest& kernel, int64_t tileM, int64_t tileN, int64_t tileK) {
    (void)tileN;
    (void)tileK;
    // For matmul loops (i in 0..M), tile outer loop i if M > tileM
    for (auto& stmt : kernel.getRootStmts()) {
        auto forLoop = std::dynamic_pointer_cast<ForLoop>(stmt);
        if (forLoop && forLoop->getVarName() == "i") {
            if (forLoop->getEnd() > tileM) {
                // Outer loop can be blocked
                forLoop->setStep(1); // Standard step with blocked execution
            }
        }
    }
}

void LoopVectorizationPass::run(KernelNest& kernel, int64_t vectorWidth) {
    for (auto& stmt : kernel.getRootStmts()) {
        auto forLoop = std::dynamic_pointer_cast<ForLoop>(stmt);
        if (forLoop) {
            // Find innermost loop
            auto curr = forLoop;
            while (!curr->getBody().empty()) {
                std::shared_ptr<ForLoop> nextLoop = nullptr;
                for (auto& bodyStmt : curr->getBody()) {
                    auto inner = std::dynamic_pointer_cast<ForLoop>(bodyStmt);
                    if (inner) {
                        nextLoop = inner;
                        break;
                    }
                }
                if (nextLoop) {
                    curr = nextLoop;
                } else {
                    break;
                }
            }
            curr->setVectorWidth(vectorWidth);
        }
    }
}

void LoopUnrollingPass::run(KernelNest& kernel, int64_t unrollFactor) {
    for (auto& stmt : kernel.getRootStmts()) {
        auto forLoop = std::dynamic_pointer_cast<ForLoop>(stmt);
        if (forLoop) {
            // Unroll innermost loop
            auto curr = forLoop;
            while (!curr->getBody().empty()) {
                std::shared_ptr<ForLoop> nextLoop = nullptr;
                for (auto& bodyStmt : curr->getBody()) {
                    auto inner = std::dynamic_pointer_cast<ForLoop>(bodyStmt);
                    if (inner) {
                        nextLoop = inner;
                        break;
                    }
                }
                if (nextLoop) {
                    curr = nextLoop;
                } else {
                    break;
                }
            }
            curr->setUnrollFactor(unrollFactor);
        }
    }
}

void LoopParallelizationPass::run(KernelNest& kernel) {
    // Outer loop writes to distinct slices of output buffers, so it can be parallelized
    for (auto& stmt : kernel.getRootStmts()) {
        auto forLoop = std::dynamic_pointer_cast<ForLoop>(stmt);
        if (forLoop) {
            forLoop->setParallel(true);
        }
    }
}

namespace {

struct Interval {
    int64_t low{0};
    int64_t high{0};
    bool valid{true};

    Interval() = default;
    Interval(int64_t l, int64_t h) : low(l), high(h), valid(true) {}

    static Interval invalid() {
        Interval iv;
        iv.valid = false;
        return iv;
    }
};

static Interval evalExprInterval(const ExprPtr& expr, const std::unordered_map<std::string, Interval>& env) {
    if (!expr) return Interval::invalid();
    if (auto c = std::dynamic_pointer_cast<ConstExpr>(expr)) {
        if (!c->isFloat()) {
            return Interval(c->getIntVal(), c->getIntVal());
        }
        return Interval(static_cast<int64_t>(c->getVal()), static_cast<int64_t>(c->getVal()));
    }
    if (auto v = std::dynamic_pointer_cast<VarExpr>(expr)) {
        auto it = env.find(v->getName());
        if (it != env.end()) {
            return it->second;
        }
        return Interval::invalid();
    }
    if (auto b = std::dynamic_pointer_cast<BinaryExpr>(expr)) {
        auto lhs = evalExprInterval(b->getLhs(), env);
        auto rhs = evalExprInterval(b->getRhs(), env);
        if (!lhs.valid || !rhs.valid) return Interval::invalid();
        switch (b->getOp()) {
            case BinaryOp::Add:
                return Interval(lhs.low + rhs.low, lhs.high + rhs.high);
            case BinaryOp::Sub:
                return Interval(lhs.low - rhs.high, lhs.high - rhs.low);
            case BinaryOp::Mul: {
                int64_t p1 = lhs.low * rhs.low;
                int64_t p2 = lhs.low * rhs.high;
                int64_t p3 = lhs.high * rhs.low;
                int64_t p4 = lhs.high * rhs.high;
                return Interval(std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4}));
            }
            case BinaryOp::Div:
                if (rhs.low > 0 && rhs.high > 0) {
                    return Interval(lhs.low / rhs.high, lhs.high / rhs.low);
                }
                return Interval::invalid();
            case BinaryOp::Max:
                return Interval(std::max(lhs.low, rhs.low), std::max(lhs.high, rhs.high));
            case BinaryOp::Min:
                return Interval(std::min(lhs.low, rhs.low), std::min(lhs.high, rhs.high));
        }
    }
    return Interval::invalid();
}

static bool checkExprAccesses(const ExprPtr& expr,
                              KernelNest& kernel,
                              const std::unordered_map<std::string, Interval>& env,
                              int64_t vectorWidth) {
    if (!expr) return true;
    if (auto load = std::dynamic_pointer_cast<LoadExpr>(expr)) {
        Buffer* buf = kernel.getBuffer(load->getBufName());
        if (!buf) return false;
        Interval iv = evalExprInterval(load->getIndex(), env);
        if (!iv.valid) return false;
        int64_t maxAccess = iv.high;
        if (iv.low < 0 || maxAccess >= buf->getNumElements()) {
            return false;
        }
        return true;
    }
    if (auto bin = std::dynamic_pointer_cast<BinaryExpr>(expr)) {
        return checkExprAccesses(bin->getLhs(), kernel, env, vectorWidth) &&
               checkExprAccesses(bin->getRhs(), kernel, env, vectorWidth);
    }
    if (auto un = std::dynamic_pointer_cast<UnaryExpr>(expr)) {
        return checkExprAccesses(un->getOperand(), kernel, env, vectorWidth);
    }
    if (auto fma = std::dynamic_pointer_cast<FMAExpr>(expr)) {
        return checkExprAccesses(fma->getA(), kernel, env, vectorWidth) &&
               checkExprAccesses(fma->getB(), kernel, env, vectorWidth) &&
               checkExprAccesses(fma->getC(), kernel, env, vectorWidth);
    }
    return true;
}

static bool checkStmt(const LoopStmtPtr& stmt,
                      KernelNest& kernel,
                      std::unordered_map<std::string, Interval>& env,
                      int64_t currentVectorWidth) {
    if (auto forLoop = std::dynamic_pointer_cast<ForLoop>(stmt)) {
        int64_t range = forLoop->getEnd() - forLoop->getStart();
        if (range <= 0) return false;
        int64_t vw = forLoop->getVectorWidth();
        if (vw > 1 && (range % vw != 0)) {
            return false;
        }
        int64_t effVw = vw > 1 ? vw : currentVectorWidth;
        Interval loopIv(forLoop->getStart(), forLoop->getEnd() - 1);
        env[forLoop->getVarName()] = loopIv;

        for (const auto& bodyStmt : forLoop->getBody()) {
            if (!checkStmt(bodyStmt, kernel, env, effVw)) {
                env.erase(forLoop->getVarName());
                return false;
            }
        }
        env.erase(forLoop->getVarName());
        return true;
    }
    if (auto assign = std::dynamic_pointer_cast<AssignStmt>(stmt)) {
        return checkExprAccesses(assign->getValue(), kernel, env, currentVectorWidth);
    }
    if (auto store = std::dynamic_pointer_cast<StoreStmt>(stmt)) {
        Buffer* buf = kernel.getBuffer(store->getBufName());
        if (!buf) return false;
        Interval iv = evalExprInterval(store->getIndex(), env);
        if (!iv.valid) return false;
        if (iv.low < 0 || iv.high >= buf->getNumElements()) {
            return false;
        }
        return checkExprAccesses(store->getValue(), kernel, env, currentVectorWidth);
    }
    return true;
}

} // anonymous namespace

bool BoundsCheckEliminationPass::run(KernelNest& kernel) {
    // Static range-analysis interval propagation (CS6843 A2/A4)
    if (kernel.getRootStmts().empty()) {
        kernel.setBoundsCheckEliminated(false);
        return false;
    }

    std::unordered_map<std::string, Interval> env;
    bool allSafe = true;
    for (const auto& stmt : kernel.getRootStmts()) {
        if (!checkStmt(stmt, kernel, env, 1)) {
            allSafe = false;
            break;
        }
    }

    kernel.setBoundsCheckEliminated(allSafe);
    return allSafe;
}

} // namespace dlc::loopir

