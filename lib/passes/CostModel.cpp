#include "dlc/passes/CostModel.h"
#include <cstdio>
#include <sstream>

namespace dlc {

namespace {

int64_t elements(const Value& value) { return static_cast<const TensorType&>(*value.getType()).getNumElements(); }

// FLOPs per output element of elementwise ops, and per input element of the row kernels.
double flopsPerElement(const std::string& name) {
    if (name == "broadcast_to" || name == "cast") return 0;
    if (name == "gelu") return 9;          // x², x³, k·x³, +, ·c, tanh, +1, ·½, ·x
    if (name == "softmax" || name == "log_softmax") return 5; // max, sub, exp, sum, div (or log)
    if (name == "layernorm") return 8;     // mean, sub, square, mean, rsqrt, mul, mul, add
    if (name == "cross_entropy") return 5; // max, sub, exp, sum, log
    return 1;
}

} // namespace

int64_t tensorBytes(const Value& value) { return static_cast<const TensorType&>(*value.getType()).getSizeBytes(); }

Cost opCost(const Operation& op) {
    Cost c;
    const std::string& name = op.getOpName();
    if (name == "return" || name == "reshape") return c;
    for (const auto& operand : op.getOperands()) c.bytesRead += tensorBytes(*operand);
    for (const auto& result : op.getResults()) c.bytesWritten += tensorBytes(*result);
    if (name == "constant" || name == "transpose" || name == "embedding" || name == "onehot") return c;
    if (name == "matmul") {
        const Shape& a = static_cast<const TensorType&>(*op.getOperand(0)->getType()).getShape();
        bool ta = op.hasAttribute("transpose_a") && op.getIntAttr("transpose_a") == 1;
        int64_t k = ta ? a[a.size() - 2] : a.back();
        c.flops = 2.0 * static_cast<double>(elements(*op.getResult(0))) * static_cast<double>(k);
    } else if (name == "sum" || name == "mean" || name == "reduce_max" || name == "scatter_add") {
        c.flops = static_cast<double>(elements(*op.getOperand(name == "scatter_add" ? 1 : 0)));
    } else if (name == "softmax" || name == "log_softmax" || name == "layernorm" || name == "cross_entropy") {
        c.flops = flopsPerElement(name) * static_cast<double>(elements(*op.getOperand(0)));
    } else {
        c.flops = flopsPerElement(name) * static_cast<double>(elements(*op.getResult(0)));
    }
    return c;
}

Cost kernelCost(const Kernel& kernel) {
    Cost c;
    for (const Operation* op : kernel.ops) c.flops += opCost(*op).flops;
    for (const Value* in : kernel.inputs) c.bytesRead += tensorBytes(*in);
    for (const Value* out : kernel.outputs) c.bytesWritten += tensorBytes(*out);
    return c;
}

std::string costReport(const Function& func, const std::vector<Kernel>& kernels) {
    std::ostringstream os;
    char line[256];
    os << "Cost model for @" << func.getName() << " (" << kernels.size() << " kernels)\n";
    std::snprintf(line, sizeof line, "  %-40s %-12s %14s %14s %10s\n", "kernel", "kind", "FLOPs", "bytes", "FLOP/byte");
    os << line;
    Cost fused;
    for (const auto& k : kernels) {
        Cost c = kernelCost(k);
        fused.flops += c.flops;
        fused.bytesRead += c.bytesRead;
        fused.bytesWritten += c.bytesWritten;
        std::snprintf(line, sizeof line, "  %-40s %-12s %14.0f %14lld %10.2f\n", k.name.c_str(),
                      kernelKindName(k.kind).c_str(), c.flops, static_cast<long long>(c.bytes()), c.intensity());
        os << line;
    }
    Cost unfused;
    for (const auto& op : func.getEntryBlock()->getOperations()) {
        if (op->getOpName() == "constant") continue; // splats are inlined either way
        Cost c = opCost(*op);
        unfused.flops += c.flops;
        unfused.bytesRead += c.bytesRead;
        unfused.bytesWritten += c.bytesWritten;
    }
    std::snprintf(line, sizeof line, "  total: %.0f FLOPs; %lld bytes fused vs %lld bytes op by op (%.1f%% less)\n",
                  fused.flops, static_cast<long long>(fused.bytes()), static_cast<long long>(unfused.bytes()),
                  unfused.bytes() == 0 ? 0.0 : 100.0 * (1.0 - static_cast<double>(fused.bytes()) / unfused.bytes()));
    os << line;
    return os.str();
}

} // namespace dlc
