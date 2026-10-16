#include "dlc/passes/Fusion.h"
#include <vector>
#include <unordered_map>

namespace dlc {

static bool isElementwiseOp(const std::string& name) {
    return name == "add" || name == "sub" || name == "mul" || name == "div" ||
           name == "relu" || name == "exp" || name == "log" || name == "gelu";
}

static bool isReductionOp(const std::string& name) {
    return name == "sum" || name == "mean" || name == "softmax" ||
           name == "log_softmax" || name == "cross_entropy";
}

bool FusionPass::runOnFunction(Function& func) {
    fusedGroups_.clear();
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    const auto& ops = entry->getOperations();
    std::unordered_set<Operation*> visited;
    int64_t currentGroupId = 0;

    // Build use-def map
    std::unordered_map<Value*, std::vector<Operation*>> uses;
    for (const auto& opPtr : ops) {
        Operation* op = opPtr.get();
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            uses[op->getOperand(i).get()].push_back(op);
        }
    }

    for (size_t i = 0; i < ops.size(); ++i) {
        Operation* op = ops[i].get();
        if (op->getOpName() == "return" || visited.count(op)) continue;

        // 1. Matmul + Epilogue Fusion (matmul -> add(bias) -> relu)
        if (op->getOpName() == "matmul" && op->getNumResults() > 0) {
            Value* mmRes = op->getResult(0).get();
            Operation* biasAdd = nullptr;
            Operation* act = nullptr;

            if (uses[mmRes].size() == 1) {
                Operation* candidateAdd = uses[mmRes][0];
                if (candidateAdd->getOpName() == "add") {
                    biasAdd = candidateAdd;
                    if (biasAdd->getNumResults() > 0 && uses[biasAdd->getResult(0).get()].size() == 1) {
                        Operation* candidateAct = uses[biasAdd->getResult(0).get()][0];
                        if (candidateAct->getOpName() == "relu") {
                            act = candidateAct;
                        }
                    }
                }
            }

            FusedGroup group;
            group.groupId = currentGroupId++;
            group.kind = FusedGroupKind::MatmulEpilogue;
            group.matmulOp = op;
            group.operations.push_back(op);
            group.opSet.insert(op);
            visited.insert(op);

            if (biasAdd) {
                group.biasAddOp = biasAdd;
                group.operations.push_back(biasAdd);
                group.opSet.insert(biasAdd);
                visited.insert(biasAdd);
            }
            if (act) {
                group.actOp = act;
                group.operations.push_back(act);
                group.opSet.insert(act);
                visited.insert(act);
            }

            if (act) {
                group.name = "fused_matmul_bias_relu";
            } else if (biasAdd) {
                group.name = "fused_matmul_bias";
            } else {
                group.name = "matmul";
            }

            fusedGroups_.push_back(std::move(group));
            continue;
        }

        // 2. Elementwise Chain Fusion
        if (isElementwiseOp(op->getOpName())) {
            FusedGroup group;
            group.groupId = currentGroupId++;
            group.kind = FusedGroupKind::Elementwise;
            group.name = "fused_elementwise_" + op->getOpName();
            group.operations.push_back(op);
            group.opSet.insert(op);
            visited.insert(op);

            // Greedily absorb subsequent dependent elementwise ops with matching shape
            Shape baseShape;
            if (op->getNumResults() > 0 && op->getResult(0)->getType()->isTensor()) {
                baseShape = std::static_pointer_cast<TensorType>(op->getResult(0)->getType())->getShape();
            }

            for (size_t j = i + 1; j < ops.size(); ++j) {
                Operation* nextOp = ops[j].get();
                if (visited.count(nextOp) || nextOp->getOpName() == "return") continue;
                if (!isElementwiseOp(nextOp->getOpName())) break;

                // Check if nextOp consumes any output of the current group
                bool consumesGroup = false;
                for (size_t opIdx = 0; opIdx < nextOp->getNumOperands(); ++opIdx) {
                    Value* operand = nextOp->getOperand(opIdx).get();
                    for (Operation* grpOp : group.operations) {
                        for (size_t resIdx = 0; resIdx < grpOp->getNumResults(); ++resIdx) {
                            if (grpOp->getResult(resIdx).get() == operand) {
                                consumesGroup = true;
                                break;
                            }
                        }
                        if (consumesGroup) break;
                    }
                    if (consumesGroup) break;
                }

                if (consumesGroup) {
                    // Check shape compatibility
                    if (nextOp->getNumResults() > 0 && nextOp->getResult(0)->getType()->isTensor()) {
                        const auto& nextShape = std::static_pointer_cast<TensorType>(nextOp->getResult(0)->getType())->getShape();
                        if (nextShape == baseShape) {
                            group.operations.push_back(nextOp);
                            group.opSet.insert(nextOp);
                            visited.insert(nextOp);
                        }
                    }
                }
            }

            fusedGroups_.push_back(std::move(group));
            continue;
        }

        // 3. Reduction ops
        if (isReductionOp(op->getOpName())) {
            FusedGroup group;
            group.groupId = currentGroupId++;
            group.kind = FusedGroupKind::Reduction;
            group.name = op->getOpName();
            group.operations.push_back(op);
            group.opSet.insert(op);
            visited.insert(op);
            fusedGroups_.push_back(std::move(group));
            continue;
        }

        // 4. Generic standalone ops
        FusedGroup group;
        group.groupId = currentGroupId++;
        group.kind = FusedGroupKind::Generic;
        group.name = op->getOpName();
        group.operations.push_back(op);
        group.opSet.insert(op);
        visited.insert(op);
    }

    // Annotate operations in IR with fusion grouping for inspection and IR dumps
    for (const auto& group : fusedGroups_) {
        for (Operation* grpOp : group.operations) {
            if (grpOp) {
                grpOp->setAttribute("fusion.group", group.groupId);
                grpOp->setAttribute("fusion.kernel", group.name);
            }
        }
    }

    return !fusedGroups_.empty();
}

} // namespace dlc

