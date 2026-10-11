#pragma once

#include "dlc/ir/Type.h"
#include <string>
#include <vector>
#include <memory>

namespace dlc::loopir {

struct Buffer {
    std::string name;
    DataType elemType;
    Shape shape;
    int64_t arenaOffsetBytes{0};
};

class LoopStmt {
public:
    virtual ~LoopStmt() = default;
    virtual std::string toString() const = 0;
};

using LoopStmtPtr = std::shared_ptr<LoopStmt>;

class ForLoop : public LoopStmt {
public:
    ForLoop(std::string varName, int64_t start, int64_t end, int64_t step = 1)
        : varName_(std::move(varName)), start_(start), end_(end), step_(step) {}

    const std::string& getVarName() const { return varName_; }
    int64_t getStart() const { return start_; }
    int64_t getEnd() const { return end_; }
    int64_t getStep() const { return step_; }

    bool isParallel() const { return parallel_; }
    void setParallel(bool p) { parallel_ = p; }

    int64_t getVectorWidth() const { return vectorWidth_; }
    void setVectorWidth(int64_t w) { vectorWidth_ = w; }

    int64_t getUnrollFactor() const { return unrollFactor_; }
    void setUnrollFactor(int64_t u) { unrollFactor_ = u; }

    void addBodyStmt(LoopStmtPtr stmt) { body_.push_back(std::move(stmt)); }
    const std::vector<LoopStmtPtr>& getBody() const { return body_; }

    std::string toString() const override;

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

    void addRootStmt(LoopStmtPtr stmt) { rootStmts_.push_back(std::move(stmt)); }
    const std::vector<LoopStmtPtr>& getRootStmts() const { return rootStmts_; }

    std::string toString() const;

private:
    std::string name_;
    std::vector<Buffer> buffers_;
    std::vector<LoopStmtPtr> rootStmts_;
};

} // namespace dlc::loopir
