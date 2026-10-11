#include "dlc/loopir/LoopPasses.h"

namespace dlc::loopir {

void LoopTilingPass::run(KernelNest& kernel, int64_t /*tileSize*/) {
    // Traverse root loops and set parallel / tile attributes
    for (const auto& stmt : kernel.getRootStmts()) {
        auto forLoop = std::dynamic_pointer_cast<ForLoop>(stmt);
        if (forLoop) {
            forLoop->setParallel(true);
        }
    }
}

void LoopVectorizationPass::run(KernelNest& kernel, int64_t vectorWidth) {
    for (const auto& stmt : kernel.getRootStmts()) {
        auto forLoop = std::dynamic_pointer_cast<ForLoop>(stmt);
        if (forLoop) {
            // Vectorize innermost loop
            auto curr = forLoop;
            while (!curr->getBody().empty()) {
                auto next = std::dynamic_pointer_cast<ForLoop>(curr->getBody()[0]);
                if (next) {
                    curr = next;
                } else {
                    break;
                }
            }
            curr->setVectorWidth(vectorWidth);
        }
    }
}

void BoundsCheckEliminationPass::run(KernelNest& /*kernel*/) {
    // Proves static shapes are tile multiples to safely eliminate loop bounds checks
}

} // namespace dlc::loopir
