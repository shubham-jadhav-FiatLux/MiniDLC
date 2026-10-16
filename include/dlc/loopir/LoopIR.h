#pragma once

#include "dlc/ir/Type.h"
#include <string>
#include <vector>
#include <memory>
#include <sstream>

namespace dlc::loopir {

// Memory buffer allocated inside the unified arena
struct Buffer {
    std::string name;
    DataType elemType{DataType::Float32};
    Shape shape;
    std::vector<int64_t> strides;
    int64_t arenaOffsetBytes{0};
    int64_t sizeBytes{0};
    bool isInput{false};
    bool isOutput{false};
    bool isDonated{false};

    int64_t getNumElements() const {
        if (shape.empty()) return 0;
        int64_t count = 1;
        for (auto dim : shape) count *= dim;
        return count;
    }

    void computeContiguousStrides() {
        strides.resize(shape.size());
        if (shape.empty()) return;
        int64_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
};

// Expression AST for loop nest bodies
class Expr {
public:
    virtual ~Expr() = default;
    virtual std::string toString() const = 0;
};
using ExprPtr = std::shared_ptr<Expr>;

class VarExpr : public Expr {
public:
    explicit VarExpr(std::string name) : name_(std::move(name)) {}
    const std::string& getName() const { return name_; }
    std::string toString() const override { return name_; }
private:
    std::string name_;
};

class ConstExpr : public Expr {
public:
    explicit ConstExpr(double val) : val_(val), isFloat_(true) {}
    explicit ConstExpr(int64_t intVal) : intVal_(intVal), isFloat_(false) {}
    explicit ConstExpr(int intVal) : intVal_(intVal), isFloat_(false) {}
    double getVal() const { return val_; }

    int64_t getIntVal() const { return intVal_; }
    bool isFloat() const { return isFloat_; }
    std::string toString() const override {
        if (isFloat_) {
            std::ostringstream ss;
            ss << val_;
            if (ss.str().find('.') == std::string::npos) ss << ".0";
            return ss.str();
        }
        return std::to_string(intVal_);
    }
private:
    double val_{0.0};
    int64_t intVal_{0};
    bool isFloat_{true};
};

class LoadExpr : public Expr {
public:
    LoadExpr(std::string bufName, ExprPtr index)
        : bufName_(std::move(bufName)), index_(std::move(index)) {}
    const std::string& getBufName() const { return bufName_; }
    const ExprPtr& getIndex() const { return index_; }
    std::string toString() const override {
        return bufName_ + "[" + (index_ ? index_->toString() : "0") + "]";
    }
private:
    std::string bufName_;
    ExprPtr index_;
};

enum class BinaryOp {
    Add, Sub, Mul, Div, Max, Min
};

class BinaryExpr : public Expr {
public:
    BinaryExpr(BinaryOp op, ExprPtr lhs, ExprPtr rhs)
        : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
    BinaryOp getOp() const { return op_; }
    const ExprPtr& getLhs() const { return lhs_; }
    const ExprPtr& getRhs() const { return rhs_; }
    std::string toString() const override;
private:
    BinaryOp op_;
    ExprPtr lhs_;
    ExprPtr rhs_;
};

enum class UnaryOp {
    Exp, Log, Relu, Sqrt, Neg
};

class UnaryExpr : public Expr {
public:
    UnaryExpr(UnaryOp op, ExprPtr operand)
        : op_(op), operand_(std::move(operand)) {}
    UnaryOp getOp() const { return op_; }
    const ExprPtr& getOperand() const { return operand_; }
    std::string toString() const override;
private:
    UnaryOp op_;
    ExprPtr operand_;
};

class FMAExpr : public Expr {
public:
    FMAExpr(ExprPtr a, ExprPtr b, ExprPtr c)
        : a_(std::move(a)), b_(std::move(b)), c_(std::move(c)) {}
    const ExprPtr& getA() const { return a_; }
    const ExprPtr& getB() const { return b_; }
    const ExprPtr& getC() const { return c_; }
    std::string toString() const override {
        return "fma(" + a_->toString() + ", " + b_->toString() + ", " + c_->toString() + ")";
    }
private:
    ExprPtr a_;
    ExprPtr b_;
    ExprPtr c_;
};

// Statement AST
class LoopStmt {
public:
    virtual ~LoopStmt() = default;
    virtual std::string toString(int indent = 0) const = 0;
};
using LoopStmtPtr = std::shared_ptr<LoopStmt>;

class AssignStmt : public LoopStmt {
public:
    AssignStmt(std::string varName, ExprPtr value)
        : varName_(std::move(varName)), value_(std::move(value)) {}
    const std::string& getVarName() const { return varName_; }
    const ExprPtr& getValue() const { return value_; }
    std::string toString(int indent = 0) const override;
private:
    std::string varName_;
    ExprPtr value_;
};

class StoreStmt : public LoopStmt {
public:
    StoreStmt(std::string bufName, ExprPtr index, ExprPtr value)
        : bufName_(std::move(bufName)), index_(std::move(index)), value_(std::move(value)) {}
    const std::string& getBufName() const { return bufName_; }
    const ExprPtr& getIndex() const { return index_; }
    const ExprPtr& getValue() const { return value_; }
    std::string toString(int indent = 0) const override;
private:
    std::string bufName_;
    ExprPtr index_;
    ExprPtr value_;
};

class ForLoop : public LoopStmt {
public:
    ForLoop(std::string varName, int64_t start, int64_t end, int64_t step = 1)
        : varName_(std::move(varName)), start_(start), end_(end), step_(step) {}

    const std::string& getVarName() const { return varName_; }
    int64_t getStart() const { return start_; }
    int64_t getEnd() const { return end_; }
    int64_t getStep() const { return step_; }

    void setStart(int64_t s) { start_ = s; }
    void setEnd(int64_t e) { end_ = e; }
    void setStep(int64_t s) { step_ = s; }

    bool isParallel() const { return parallel_; }
    void setParallel(bool p) { parallel_ = p; }

    int64_t getVectorWidth() const { return vectorWidth_; }
    void setVectorWidth(int64_t w) { vectorWidth_ = w; }

    int64_t getUnrollFactor() const { return unrollFactor_; }
    void setUnrollFactor(int64_t u) { unrollFactor_ = u; }

    void addBodyStmt(LoopStmtPtr stmt) { body_.push_back(std::move(stmt)); }
    const std::vector<LoopStmtPtr>& getBody() const { return body_; }
    std::vector<LoopStmtPtr>& getBody() { return body_; }

    std::string toString(int indent = 0) const override;

private:
    std::string varName_;
    int64_t start_;
    int64_t end_;
    int64_t step_;
    bool parallel_{false};
    int64_t vectorWidth_{1};
    int64_t unrollFactor_{1};
    std::vector<LoopStmtPtr> body_;
};

class KernelNest {
public:
    explicit KernelNest(std::string name) : name_(std::move(name)) {}

    const std::string& getName() const { return name_; }
    void addBuffer(Buffer buf) { buffers_.push_back(std::move(buf)); }
    const std::vector<Buffer>& getBuffers() const { return buffers_; }
    std::vector<Buffer>& getBuffers() { return buffers_; }

    Buffer* getBuffer(const std::string& name) {
        for (auto& b : buffers_) {
            if (b.name == name) return &b;
        }
        return nullptr;
    }

    void addRootStmt(LoopStmtPtr stmt) { rootStmts_.push_back(std::move(stmt)); }
    const std::vector<LoopStmtPtr>& getRootStmts() const { return rootStmts_; }
    std::vector<LoopStmtPtr>& getRootStmts() { return rootStmts_; }

    int64_t getEstimatedFlops() const { return estimatedFlops_; }
    void setEstimatedFlops(int64_t flops) { estimatedFlops_ = flops; }

    int64_t getEstimatedBytesMoved() const { return estimatedBytesMoved_; }
    void setEstimatedBytesMoved(int64_t bytes) { estimatedBytesMoved_ = bytes; }

    std::string toString() const;

private:
    std::string name_;
    std::vector<Buffer> buffers_;
    std::vector<LoopStmtPtr> rootStmts_;
    int64_t estimatedFlops_{0};
    int64_t estimatedBytesMoved_{0};
};

} // namespace dlc::loopir
