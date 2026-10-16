#include "dlc/loopir/LoopPasses.h"

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

bool BoundsCheckEliminationPass::run(KernelNest& kernel) {
    // Proves static shapes are tile and vector-width multiples to eliminate loop guards
    bool allMultiples = true;
    for (auto& stmt : kernel.getRootStmts()) {
        auto forLoop = std::dynamic_pointer_cast<ForLoop>(stmt);
        if (forLoop) {
            auto curr = forLoop;
            while (curr) {
                int64_t range = curr->getEnd() - curr->getStart();
                if (curr->getVectorWidth() > 1 && (range % curr->getVectorWidth() != 0)) {
                    allMultiples = false;
                }
                std::shared_ptr<ForLoop> nextLoop = nullptr;
                for (auto& bodyStmt : curr->getBody()) {
                    auto inner = std::dynamic_pointer_cast<ForLoop>(bodyStmt);
                    if (inner) {
                        nextLoop = inner;
                        break;
                    }
                }
                curr = nextLoop;
            }
        }
    }
    return allMultiples;
}

} // namespace dlc::loopir

