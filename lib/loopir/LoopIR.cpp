#include "dlc/loopir/LoopIR.h"
#include <sstream>

namespace dlc::loopir {

std::string ForLoop::toString() const {
    std::ostringstream oss;
    if (parallel_) oss << "parallel ";
    oss << "for " << varName_ << " in " << start_ << ".." << end_;
    if (step_ != 1) oss << " step " << step_;
    if (vectorWidth_ > 1) oss << " vectorize(" << vectorWidth_ << ")";
    if (unrollFactor_ > 1) oss << " unroll(" << unrollFactor_ << ")";
    oss << " {\n";
    for (const auto& s : body_) {
        oss << "  " << s->toString() << "\n";
    }
    oss << "}";
    return oss.str();
}

std::string KernelNest::toString() const {
    std::ostringstream oss;
    oss << "kernel @" << name_ << " {\n";
    for (const auto& b : buffers_) {
        oss << "  buffer " << b.name << " : offset " << b.arenaOffsetBytes << "\n";
    }
    for (const auto& s : rootStmts_) {
        oss << "  " << s->toString() << "\n";
    }
    oss << "}\n";
    return oss.str();
}

} // namespace dlc::loopir
