#include "dlc/passes/ShapeInference.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/OpRegistry.h"

namespace dlc {

bool ShapeInferencePass::broadcastShapes(const Shape& a, const Shape& b, Shape& outShape) {
    return computeBroadcastShape(a, b, outShape);
}

bool ShapeInferencePass::runOnFunction(Function& func) {
    bool changed = false;
    Block* entry = func.getEntryBlock();
    if (!entry) return false;

    for (auto& op : entry->getOperations()) {
        const OpDef* def = OpRegistry::instance().lookup(op->getOpName());
        if (!def) throw IRError("unknown op '" + op->getOpName() + "'");
        if (!def->hasResult) continue;

        TypePtr inferred;
        try {
            inferred = def->inferType(*op);
        } catch (const IRError& e) {
            throw IRError("'" + op->toString() + "': " + e.what());
        }
        ValuePtr res = op->getResult(0);
        if (!res->getType() || res->getType()->toString() != inferred->toString()) {
            res->setType(inferred);
            changed = true;
        }
    }

    return changed;
}

} // namespace dlc
