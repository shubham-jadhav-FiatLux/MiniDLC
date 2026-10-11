#pragma once

#include "dlc/ir/Operation.h"

namespace dlc {

// Linear algebra
class MatmulOp : public Operation {
public:
    MatmulOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType);
};

// Binary elementwise
class AddOp : public Operation {
public:
    AddOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType);
};

class SubOp : public Operation {
public:
    SubOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType);
};

class MulOp : public Operation {
public:
    MulOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType);
};

class DivOp : public Operation {
public:
    DivOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType);
};

// Unary elementwise
class ReluOp : public Operation {
public:
    ReluOp(ValuePtr operand, TypePtr resultType);
};

class GeluOp : public Operation {
public:
    GeluOp(ValuePtr operand, TypePtr resultType);
};

class ExpOp : public Operation {
public:
    ExpOp(ValuePtr operand, TypePtr resultType);
};

class LogOp : public Operation {
public:
    LogOp(ValuePtr operand, TypePtr resultType);
};

// Normalization & Reductions
class SoftmaxOp : public Operation {
public:
    SoftmaxOp(ValuePtr operand, int64_t axis, TypePtr resultType);
    int64_t getAxis() const;
};

class LogSoftmaxOp : public Operation {
public:
    LogSoftmaxOp(ValuePtr operand, int64_t axis, TypePtr resultType);
    int64_t getAxis() const;
};

class CrossEntropyOp : public Operation {
public:
    CrossEntropyOp(ValuePtr logits, ValuePtr targets, TypePtr resultType);
};

class LayerNormOp : public Operation {
public:
    LayerNormOp(ValuePtr x, ValuePtr gamma, ValuePtr beta, double eps, TypePtr resultType);
    double getEps() const;
};

class SumOp : public Operation {
public:
    SumOp(ValuePtr operand, int64_t axis, bool keepdims, TypePtr resultType);
    int64_t getAxis() const;
    bool getKeepDims() const;
};

class MeanOp : public Operation {
public:
    MeanOp(ValuePtr operand, int64_t axis, bool keepdims, TypePtr resultType);
    int64_t getAxis() const;
    bool getKeepDims() const;
};

// Shape & Indexing
class TransposeOp : public Operation {
public:
    TransposeOp(ValuePtr operand, std::vector<int64_t> permutation, TypePtr resultType);
    std::vector<int64_t> getPermutation() const;
};

class ReshapeOp : public Operation {
public:
    ReshapeOp(ValuePtr operand, Shape newShape, TypePtr resultType);
    Shape getNewShape() const;
};

class EmbeddingOp : public Operation {
public:
    EmbeddingOp(ValuePtr table, ValuePtr indices, TypePtr resultType);
};

class ConstantOp : public Operation {
public:
    ConstantOp(double val, TypePtr resultType);
    double getValue() const;
};

class OneHotOp : public Operation {
public:
    OneHotOp(ValuePtr indices, int64_t numClasses, TypePtr resultType);
    int64_t getNumClasses() const;
};

// Control / Terminator
class ReturnOp : public Operation {
public:
    explicit ReturnOp(std::vector<ValuePtr> returnValues);
};

} // namespace dlc
