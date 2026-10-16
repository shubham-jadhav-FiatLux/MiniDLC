#include "dlc/loopir/Lowering.h"
#include "dlc/ir/Ops.h"
#include <iostream>

namespace dlc::loopir {

static Buffer makeBufferFromValue(Value* val, const MemoryPlanningPass* memPlan, bool isInput = false, bool isOutput = false) {
    Buffer buf;
    buf.name = val->getName();
    buf.isInput = isInput;
    buf.isOutput = isOutput;
    if (val->getType()->isTensor()) {
        auto tt = std::static_pointer_cast<TensorType>(val->getType());
        buf.elemType = tt->getElementType();
        buf.shape = tt->getShape();
        buf.sizeBytes = tt->getSizeBytes();
        buf.computeContiguousStrides();
    }
    if (memPlan) {
        const auto& offsets = memPlan->getOffsets();
        auto it = offsets.find(val);
        if (it != offsets.end()) {
            buf.arenaOffsetBytes = it->second.offsetBytes;
            buf.sizeBytes = it->second.sizeBytes;
        }
    }
    return buf;
}

static KernelNest lowerMatmulEpilogueGroup(
    const FusedGroup& group,
    const MemoryPlanningPass* memPlan
) {
    Operation* mmOp = group.matmulOp;
    Operation* biasAdd = group.biasAddOp;
    Operation* act = group.actOp;

    Value* valA = mmOp->getOperand(0).get();
    Value* valB = mmOp->getOperand(1).get();

    auto lhsType = std::static_pointer_cast<TensorType>(valA->getType());
    auto rhsType = std::static_pointer_cast<TensorType>(valB->getType());
    int64_t M = lhsType->getShape()[0];
    int64_t K = lhsType->getShape()[1];
    int64_t N = rhsType->getShape()[1];

    Value* finalOut = act ? act->getResult(0).get() :
                     (biasAdd ? biasAdd->getResult(0).get() : mmOp->getResult(0).get());

    KernelNest kernel(group.name);

    Buffer bufA = makeBufferFromValue(valA, memPlan, true, false);
    Buffer bufB = makeBufferFromValue(valB, memPlan, true, false);
    kernel.addBuffer(bufA);
    kernel.addBuffer(bufB);

    Value* valBias = nullptr;
    if (biasAdd) {
        // Find which operand of biasAdd is not mmOp's result
        Value* mmRes = mmOp->getResult(0).get();
        if (biasAdd->getOperand(0).get() == mmRes) {
            valBias = biasAdd->getOperand(1).get();
        } else {
            valBias = biasAdd->getOperand(0).get();
        }
        kernel.addBuffer(makeBufferFromValue(valBias, memPlan, true, false));
    }

    Buffer bufOut = makeBufferFromValue(finalOut, memPlan, false, true);
    kernel.addBuffer(bufOut);

    // Compute FLOPs & bytes moved
    int64_t flops = 2 * M * N * K + (biasAdd ? M * N : 0) + (act ? M * N : 0);
    int64_t bytesMoved = (M * K + K * N + (valBias ? N : 0) + M * N) * sizeof(float);
    kernel.setEstimatedFlops(flops);
    kernel.setEstimatedBytesMoved(bytesMoved);

    // Loop Nest AST:
    // for i in 0..M
    //   for j in 0..N
    //     acc = 0.0
    //     for k in 0..K
    //       acc = fma(A[i*K + k], B[k*N + j], acc)
    //     if bias: acc = acc + bias[j]
    //     if relu: acc = max(acc, 0.0)
    //     Out[i*N + j] = acc
    auto loopI = std::make_shared<ForLoop>("i", 0, M);
    auto loopJ = std::make_shared<ForLoop>("j", 0, N);
    auto loopK = std::make_shared<ForLoop>("k", 0, K);

    // acc = fma(A[i*K + k], B[k*N + j], acc)
    // index expr for A: i * K + k
    auto idxA = std::make_shared<BinaryExpr>(
        BinaryOp::Add,
        std::make_shared<BinaryExpr>(BinaryOp::Mul, std::make_shared<VarExpr>("i"), std::make_shared<ConstExpr>(K)),
        std::make_shared<VarExpr>("k")
    );
    // index expr for B: k * N + j
    auto idxB = std::make_shared<BinaryExpr>(
        BinaryOp::Add,
        std::make_shared<BinaryExpr>(BinaryOp::Mul, std::make_shared<VarExpr>("k"), std::make_shared<ConstExpr>(N)),
        std::make_shared<VarExpr>("j")
    );

    auto loadA = std::make_shared<LoadExpr>(bufA.name, idxA);
    auto loadB = std::make_shared<LoadExpr>(bufB.name, idxB);
    auto fmaAcc = std::make_shared<FMAExpr>(loadA, loadB, std::make_shared<VarExpr>("acc"));
    loopK->addBodyStmt(std::make_shared<AssignStmt>("acc", fmaAcc));

    // loopJ body:
    loopJ->addBodyStmt(std::make_shared<AssignStmt>("acc", std::make_shared<ConstExpr>(0.0)));
    loopJ->addBodyStmt(loopK);

    ExprPtr epilogueVal = std::make_shared<VarExpr>("acc");
    if (valBias) {
        auto loadBias = std::make_shared<LoadExpr>(valBias->getName(), std::make_shared<VarExpr>("j"));
        epilogueVal = std::make_shared<BinaryExpr>(BinaryOp::Add, epilogueVal, loadBias);
    }
    if (act && act->getOpName() == "relu") {
        epilogueVal = std::make_shared<BinaryExpr>(BinaryOp::Max, epilogueVal, std::make_shared<ConstExpr>(0.0));
    }

    auto idxOut = std::make_shared<BinaryExpr>(
        BinaryOp::Add,
        std::make_shared<BinaryExpr>(BinaryOp::Mul, std::make_shared<VarExpr>("i"), std::make_shared<ConstExpr>(N)),
        std::make_shared<VarExpr>("j")
    );
    loopJ->addBodyStmt(std::make_shared<StoreStmt>(bufOut.name, idxOut, epilogueVal));

    loopI->addBodyStmt(loopJ);
    kernel.addRootStmt(loopI);

    return kernel;
}

static KernelNest lowerElementwiseGroup(
    const FusedGroup& group,
    const MemoryPlanningPass* memPlan
) {
    KernelNest kernel(group.name);
    Operation* firstOp = group.operations.front();
    Operation* lastOp = group.operations.back();

    auto outType = std::static_pointer_cast<TensorType>(lastOp->getResult(0)->getType());
    int64_t totalElements = outType->getNumElements();

    // Register all distinct input buffers
    std::unordered_set<std::string> registered;
    for (Operation* op : group.operations) {
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            Value* opVal = op->getOperand(i).get();
            // If operand is not produced inside the group, it's an external input
            bool internal = false;
            for (Operation* grpOp : group.operations) {
                for (size_t r = 0; r < grpOp->getNumResults(); ++r) {
                    if (grpOp->getResult(r).get() == opVal) {
                        internal = true;
                        break;
                    }
                }
                if (internal) break;
            }
            if (!internal && !registered.count(opVal->getName())) {
                kernel.addBuffer(makeBufferFromValue(opVal, memPlan, true, false));
                registered.insert(opVal->getName());
            }
        }
    }

    Buffer outBuf = makeBufferFromValue(lastOp->getResult(0).get(), memPlan, false, true);
    kernel.addBuffer(outBuf);

    int64_t flops = static_cast<int64_t>(group.operations.size()) * totalElements;
    int64_t bytesMoved = (static_cast<int64_t>(kernel.getBuffers().size())) * totalElements * sizeof(float);
    kernel.setEstimatedFlops(flops);
    kernel.setEstimatedBytesMoved(bytesMoved);

    // 1D Loop Nest: for idx in 0..totalElements
    auto loopIdx = std::make_shared<ForLoop>("idx", 0, totalElements);
    auto idxVar = std::make_shared<VarExpr>("idx");

    // Build chained expression
    // For simple chain: op0 -> op1
    ExprPtr currentExpr = nullptr;
    if (firstOp->getNumOperands() == 2) {
        BinaryOp bop = BinaryOp::Add;
        if (firstOp->getOpName() == "sub") bop = BinaryOp::Sub;
        else if (firstOp->getOpName() == "mul") bop = BinaryOp::Mul;
        else if (firstOp->getOpName() == "div") bop = BinaryOp::Div;

        // Check if second operand is broadcastable (e.g. 1D bias or scalar)
        auto op0 = firstOp->getOperand(0).get();
        auto op1 = firstOp->getOperand(1).get();
        auto op0Type = std::static_pointer_cast<TensorType>(op0->getType());
        auto op1Type = std::static_pointer_cast<TensorType>(op1->getType());

        ExprPtr load0 = std::make_shared<LoadExpr>(op0->getName(), idxVar);
        ExprPtr load1;
        if (op1Type->getNumElements() == 1) {
            load1 = std::make_shared<LoadExpr>(op1->getName(), std::make_shared<ConstExpr>(0));
        } else if (op1Type->getNumElements() < op0Type->getNumElements() && !op1Type->getShape().empty()) {
            int64_t innerDim = op1Type->getShape().back();
            auto modIdx = std::make_shared<BinaryExpr>(BinaryOp::Sub, idxVar,
                std::make_shared<BinaryExpr>(BinaryOp::Mul,
                    std::make_shared<BinaryExpr>(BinaryOp::Div, idxVar, std::make_shared<ConstExpr>(innerDim)),
                    std::make_shared<ConstExpr>(innerDim)));
            load1 = std::make_shared<LoadExpr>(op1->getName(), modIdx);
        } else {
            load1 = std::make_shared<LoadExpr>(op1->getName(), idxVar);
        }

        currentExpr = std::make_shared<BinaryExpr>(bop, load0, load1);
    } else if (firstOp->getNumOperands() == 1) {
        UnaryOp uop = UnaryOp::Relu;
        if (firstOp->getOpName() == "exp") uop = UnaryOp::Exp;
        else if (firstOp->getOpName() == "log") uop = UnaryOp::Log;
        auto op0 = firstOp->getOperand(0).get();
        auto load0 = std::make_shared<LoadExpr>(op0->getName(), idxVar);
        currentExpr = std::make_shared<UnaryExpr>(uop, load0);
    }

    // Apply subsequent ops in the chain
    for (size_t i = 1; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        if (op->getOpName() == "relu") {
            currentExpr = std::make_shared<BinaryExpr>(BinaryOp::Max, currentExpr, std::make_shared<ConstExpr>(0.0));
        } else if (op->getOpName() == "exp") {
            currentExpr = std::make_shared<UnaryExpr>(UnaryOp::Exp, currentExpr);
        } else if (op->getOpName() == "log") {
            currentExpr = std::make_shared<UnaryExpr>(UnaryOp::Log, currentExpr);
        }
    }

    loopIdx->addBodyStmt(std::make_shared<StoreStmt>(outBuf.name, idxVar, currentExpr));
    kernel.addRootStmt(loopIdx);

    return kernel;
}

static KernelNest lowerGenericOp(
    Operation* op,
    const MemoryPlanningPass* memPlan
) {
    KernelNest kernel(op->getOpName() + "_kernel");

    for (size_t i = 0; i < op->getNumOperands(); ++i) {
        kernel.addBuffer(makeBufferFromValue(op->getOperand(i).get(), memPlan, true, false));
    }
    for (size_t i = 0; i < op->getNumResults(); ++i) {
        kernel.addBuffer(makeBufferFromValue(op->getResult(i).get(), memPlan, false, true));
    }

    if (op->getNumResults() > 0 && op->getResult(0)->getType()->isTensor()) {
        auto resType = std::static_pointer_cast<TensorType>(op->getResult(0)->getType());
        int64_t totalElements = resType->getNumElements();
        auto loop = std::make_shared<ForLoop>("idx", 0, totalElements);
        auto idxVar = std::make_shared<VarExpr>("idx");

        ExprPtr valExpr;
        if (op->getNumOperands() >= 1) {
            valExpr = std::make_shared<LoadExpr>(op->getOperand(0)->getName(), idxVar);
        } else {
            valExpr = std::make_shared<ConstExpr>(0.0);
        }
        loop->addBodyStmt(std::make_shared<StoreStmt>(op->getResult(0)->getName(), idxVar, valExpr));
        kernel.addRootStmt(loop);
    }

    return kernel;
}

std::vector<KernelNest> Lowering::lowerFunction(
    const Function& func,
    const FusionPass* fusion,
    const MemoryPlanningPass* memPlan
) {
    std::vector<KernelNest> kernels;
    const Block* entry = func.getEntryBlock();
    if (!entry) return kernels;

    if (fusion && !fusion->getFusedGroups().empty()) {
        for (const auto& group : fusion->getFusedGroups()) {
            if (group.kind == FusedGroupKind::MatmulEpilogue) {
                kernels.push_back(lowerMatmulEpilogueGroup(group, memPlan));
            } else if (group.kind == FusedGroupKind::Elementwise) {
                kernels.push_back(lowerElementwiseGroup(group, memPlan));
            } else {
                for (Operation* op : group.operations) {
                    kernels.push_back(lowerGenericOp(op, memPlan));
                }
            }
        }
    } else {
        // Fallback: lower op by op
        for (const auto& opPtr : entry->getOperations()) {
            Operation* op = opPtr.get();
            if (op->getOpName() == "return") continue;

            if (op->getOpName() == "matmul" && op->getNumOperands() >= 2) {
                FusedGroup mmGroup;
                mmGroup.kind = FusedGroupKind::MatmulEpilogue;
                mmGroup.name = "matmul";
                mmGroup.matmulOp = op;
                mmGroup.operations.push_back(op);
                kernels.push_back(lowerMatmulEpilogueGroup(mmGroup, memPlan));
            } else {
                kernels.push_back(lowerGenericOp(op, memPlan));
            }
        }
    }

    return kernels;
}

} // namespace dlc::loopir

