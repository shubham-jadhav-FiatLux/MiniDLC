#include "dlc/loopir/LoopIR.h"
#include <sstream>

namespace dlc::loopir {

static std::string makeIndent(int indent) {
    return std::string(indent * 2, ' ');
}

std::string BinaryExpr::toString() const {
    std::string opStr;
    switch (op_) {
        case BinaryOp::Add: opStr = "+"; break;
        case BinaryOp::Sub: opStr = "-"; break;
        case BinaryOp::Mul: opStr = "*"; break;
        case BinaryOp::Div: opStr = "/"; break;
        case BinaryOp::Max: return "max(" + (lhs_ ? lhs_->toString() : "0") + ", " + (rhs_ ? rhs_->toString() : "0") + ")";
        case BinaryOp::Min: return "min(" + (lhs_ ? lhs_->toString() : "0") + ", " + (rhs_ ? rhs_->toString() : "0") + ")";
    }
    return "(" + (lhs_ ? lhs_->toString() : "0") + " " + opStr + " " + (rhs_ ? rhs_->toString() : "0") + ")";
}

std::string UnaryExpr::toString() const {
    std::string opStr;
    switch (op_) {
        case UnaryOp::Exp: opStr = "exp"; break;
        case UnaryOp::Log: opStr = "log"; break;
        case UnaryOp::Relu: opStr = "relu"; break;
        case UnaryOp::Sqrt: opStr = "sqrt"; break;
        case UnaryOp::Neg: opStr = "-"; break;
    }
    return opStr + "(" + (operand_ ? operand_->toString() : "0") + ")";
}

std::string AssignStmt::toString(int indent) const {
    return makeIndent(indent) + varName_ + " = " + (value_ ? value_->toString() : "0") + ";";
}

std::string StoreStmt::toString(int indent) const {
    return makeIndent(indent) + bufName_ + "[" + (index_ ? index_->toString() : "0") + "] = " +
           (value_ ? value_->toString() : "0") + ";";
}

std::string ForLoop::toString(int indent) const {
    std::ostringstream oss;
    oss << makeIndent(indent);
    if (parallel_) oss << "parallel ";
    oss << "for " << varName_ << " in " << start_ << ".." << end_;
    if (step_ != 1) oss << " step " << step_;
    if (vectorWidth_ > 1) oss << " vectorize(" << vectorWidth_ << ")";
    if (unrollFactor_ > 1) oss << " unroll(" << unrollFactor_ << ")";
    oss << " {\n";
    for (const auto& s : body_) {
        oss << s->toString(indent + 1) << "\n";
    }
    oss << makeIndent(indent) << "}";
    return oss.str();
}

std::string KernelNest::toString() const {
    std::ostringstream oss;
    oss << "kernel @" << name_ << " {\n";
    for (const auto& b : buffers_) {
        oss << "  buffer %" << b.name << " : offset " << b.arenaOffsetBytes
            << " bytes, size " << b.sizeBytes << " bytes, shape [";
        for (size_t i = 0; i < b.shape.size(); ++i) {
            oss << b.shape[i] << (i + 1 < b.shape.size() ? ", " : "");
        }
        oss << "]\n";
    }
    for (const auto& s : rootStmts_) {
        oss << s->toString(1) << "\n";
    }
    oss << "}\n";
    return oss.str();
}

} // namespace dlc::loopir

