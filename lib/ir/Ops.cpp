#include "dlc/ir/Ops.h"

namespace dlc {

MatmulOp::MatmulOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType)
    : Operation("matmul", {lhs, rhs}, {resultType}) {}

AddOp::AddOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType)
    : Operation("add", {lhs, rhs}, {resultType}) {}

SubOp::SubOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType)
    : Operation("sub", {lhs, rhs}, {resultType}) {}

MulOp::MulOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType)
    : Operation("mul", {lhs, rhs}, {resultType}) {}

DivOp::DivOp(ValuePtr lhs, ValuePtr rhs, TypePtr resultType)
    : Operation("div", {lhs, rhs}, {resultType}) {}

ReluOp::ReluOp(ValuePtr operand, TypePtr resultType)
    : Operation("relu", {operand}, {resultType}) {}

GeluOp::GeluOp(ValuePtr operand, TypePtr resultType)
    : Operation("gelu", {operand}, {resultType}) {}

ExpOp::ExpOp(ValuePtr operand, TypePtr resultType)
    : Operation("exp", {operand}, {resultType}) {}

LogOp::LogOp(ValuePtr operand, TypePtr resultType)
    : Operation("log", {operand}, {resultType}) {}

SoftmaxOp::SoftmaxOp(ValuePtr operand, int64_t axis, TypePtr resultType)
    : Operation("softmax", {operand}, {resultType}) {
    setAttribute("axis", axis);
}

int64_t SoftmaxOp::getAxis() const {
    return std::get<int64_t>(getAttribute("axis"));
}

LogSoftmaxOp::LogSoftmaxOp(ValuePtr operand, int64_t axis, TypePtr resultType)
    : Operation("log_softmax", {operand}, {resultType}) {
    setAttribute("axis", axis);
}

int64_t LogSoftmaxOp::getAxis() const {
    return std::get<int64_t>(getAttribute("axis"));
}

CrossEntropyOp::CrossEntropyOp(ValuePtr logits, ValuePtr targets, TypePtr resultType)
    : Operation("cross_entropy", {logits, targets}, {resultType}) {}

LayerNormOp::LayerNormOp(ValuePtr x, ValuePtr gamma, ValuePtr beta, double eps, TypePtr resultType)
    : Operation("layernorm", {x, gamma, beta}, {resultType}) {
    setAttribute("eps", eps);
}

double LayerNormOp::getEps() const {
    return std::get<double>(getAttribute("eps"));
}

SumOp::SumOp(ValuePtr operand, int64_t axis, bool keepdims, TypePtr resultType)
    : Operation("sum", {operand}, {resultType}) {
    setAttribute("axis", axis);
    setAttribute("keepdims", static_cast<int64_t>(keepdims ? 1 : 0));
}

int64_t SumOp::getAxis() const {
    return std::get<int64_t>(getAttribute("axis"));
}

bool SumOp::getKeepDims() const {
    return std::get<int64_t>(getAttribute("keepdims")) != 0;
}

MeanOp::MeanOp(ValuePtr operand, int64_t axis, bool keepdims, TypePtr resultType)
    : Operation("mean", {operand}, {resultType}) {
    setAttribute("axis", axis);
    setAttribute("keepdims", static_cast<int64_t>(keepdims ? 1 : 0));
}

int64_t MeanOp::getAxis() const {
    return std::get<int64_t>(getAttribute("axis"));
}

bool MeanOp::getKeepDims() const {
    return std::get<int64_t>(getAttribute("keepdims")) != 0;
}

TransposeOp::TransposeOp(ValuePtr operand, std::vector<int64_t> permutation, TypePtr resultType)
    : Operation("transpose", {operand}, {resultType}) {
    setAttribute("perm", permutation);
}

std::vector<int64_t> TransposeOp::getPermutation() const {
    return std::get<std::vector<int64_t>>(getAttribute("perm"));
}

ReshapeOp::ReshapeOp(ValuePtr operand, Shape newShape, TypePtr resultType)
    : Operation("reshape", {operand}, {resultType}) {
    setAttribute("shape", newShape);
}

Shape ReshapeOp::getNewShape() const {
    return std::get<Shape>(getAttribute("shape"));
}

EmbeddingOp::EmbeddingOp(ValuePtr table, ValuePtr indices, TypePtr resultType)
    : Operation("embedding", {table, indices}, {resultType}) {}

ConstantOp::ConstantOp(double val, TypePtr resultType)
    : Operation("constant", {}, {resultType}) {
    setAttribute("value", val);
}

double ConstantOp::getValue() const {
    return std::get<double>(getAttribute("value"));
}

OneHotOp::OneHotOp(ValuePtr indices, int64_t numClasses, TypePtr resultType)
    : Operation("onehot", {indices}, {resultType}) {
    setAttribute("num_classes", numClasses);
}

int64_t OneHotOp::getNumClasses() const {
    return std::get<int64_t>(getAttribute("num_classes"));
}

ReturnOp::ReturnOp(std::vector<ValuePtr> returnValues)
    : Operation("return", std::move(returnValues), {}) {}

} // namespace dlc
