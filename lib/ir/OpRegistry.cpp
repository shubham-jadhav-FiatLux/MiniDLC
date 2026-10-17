#include "dlc/ir/OpRegistry.h"
#include "dlc/ir/Error.h"
#include <algorithm>
#include <numeric>

namespace dlc {

namespace {

std::string shapeString(const Shape& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

const TensorType& operandType(const Operation& op, size_t idx) {
    if (idx >= op.getNumOperands() || !op.getOperand(idx)) {
        throw IRError("'" + op.getOpName() + "' is missing operand " + std::to_string(idx));
    }
    const TypePtr& type = op.getOperand(idx)->getType();
    if (!type || !type->isTensor()) {
        throw IRError("operand " + std::to_string(idx) + " of '" + op.getOpName() + "' has no tensor type");
    }
    return static_cast<const TensorType&>(*type);
}

const TensorType& operandOfType(const Operation& op, size_t idx, DataType expected) {
    const TensorType& type = operandType(op, idx);
    if (type.getElementType() != expected) {
        throw IRError("operand " + std::to_string(idx) + " of '" + op.getOpName() + "' must be " +
                      dataTypeToString(expected) + ", got " + type.toString());
    }
    return type;
}

const TensorType& floatOperand(const Operation& op, size_t idx) { return operandOfType(op, idx, DataType::Float32); }
const TensorType& intOperand(const Operation& op, size_t idx) { return operandOfType(op, idx, DataType::Int32); }

TypePtr tensorType(DataType dt, Shape shape) {
    if (checkedNumElements(shape) < 0) throw IRError("result shape " + shapeString(shape) + " has too many elements");
    return std::make_shared<TensorType>(dt, std::move(shape));
}

int64_t normalizeAxis(const Operation& op, int64_t axis, int64_t rank) {
    if (axis < -rank || axis >= rank) {
        throw IRError("axis " + std::to_string(axis) + " of '" + op.getOpName() + "' is out of range for rank " +
                      std::to_string(rank));
    }
    return axis < 0 ? axis + rank : axis;
}

TypePtr inferBinary(const Operation& op) {
    const TensorType& a = floatOperand(op, 0);
    const TensorType& b = floatOperand(op, 1);
    Shape out;
    if (!computeBroadcastShape(a.getShape(), b.getShape(), out)) {
        throw IRError("operands of '" + op.getOpName() + "' are not broadcast-compatible: " + a.toString() + " and " +
                      b.toString());
    }
    return tensorType(DataType::Float32, out);
}

TypePtr inferUnary(const Operation& op) { return tensorType(DataType::Float32, floatOperand(op, 0).getShape()); }

// Broadcast of any number of f32 operands (cmp_gt, select).
TypePtr inferBroadcastAll(const Operation& op) {
    Shape out;
    for (size_t i = 0; i < op.getNumOperands(); ++i) {
        const TensorType& t = floatOperand(op, i);
        Shape next;
        if (!computeBroadcastShape(out, t.getShape(), next)) {
            throw IRError("operands of '" + op.getOpName() + "' are not broadcast-compatible: operand " +
                          std::to_string(i) + " is " + t.toString());
        }
        out = std::move(next);
    }
    return tensorType(DataType::Float32, out);
}

bool flag(const Operation& op, const char* name) {
    if (!op.hasAttribute(name)) return false;
    int64_t value = op.getIntAttr(name);
    if (value != 0 && value != 1) throw IRError(std::string(name) + " of '" + op.getOpName() + "' must be 0 or 1");
    return value == 1;
}

// The shape of a matmul operand after its transpose flag (swaps the last two dimensions).
Shape matmulOperandShape(const TensorType& t, bool transposed) {
    Shape s = t.getShape();
    if (transposed && s.size() >= 2) std::swap(s[s.size() - 1], s[s.size() - 2]);
    return s;
}

TypePtr inferAxisNormalization(const Operation& op) {
    const TensorType& x = floatOperand(op, 0);
    if (x.getRank() == 0) throw IRError("'" + op.getOpName() + "' needs an operand of rank >= 1");
    normalizeAxis(op, op.getIntAttr("axis"), x.getRank());
    return tensorType(DataType::Float32, x.getShape());
}

TypePtr inferMatmul(const Operation& op) {
    const TensorType& a = floatOperand(op, 0);
    const TensorType& b = floatOperand(op, 1);
    bool ta = flag(op, "transpose_a");
    bool tb = flag(op, "transpose_b");
    if ((ta && a.getRank() < 2) || (tb && b.getRank() < 2)) throw IRError("matmul transpose flags need rank >= 2");
    const Shape sa = matmulOperandShape(a, ta);
    const Shape sb = matmulOperandShape(b, tb);
    auto mismatch = [&] {
        return IRError("matmul operand shapes don't match: " + a.toString() + (ta ? "^T" : "") + " x " + b.toString() +
                       (tb ? "^T" : ""));
    };

    if (sa.size() == 2 && sb.size() == 2) {
        if (sa[1] != sb[0]) throw mismatch();
        return tensorType(DataType::Float32, {sa[0], sb[1]});
    }
    if (sa.size() == 3 && sb.size() == 2) {
        if (sa[2] != sb[0]) throw mismatch();
        return tensorType(DataType::Float32, {sa[0], sa[1], sb[1]});
    }
    if (sa.size() == sb.size() && (sa.size() == 3 || sa.size() == 4)) {
        size_t rank = sa.size();
        if (!std::equal(sa.begin(), sa.end() - 2, sb.begin()) || sa[rank - 1] != sb[rank - 2]) throw mismatch();
        Shape out(sa.begin(), sa.end() - 1);
        out.push_back(sb[rank - 1]);
        return tensorType(DataType::Float32, out);
    }
    throw IRError("matmul supports 2-D x 2-D, 3-D x 2-D, 3-D x 3-D and 4-D x 4-D operands, got " + a.toString() +
                  " x " + b.toString());
}

TypePtr inferCrossEntropy(const Operation& op) {
    const TensorType& logits = floatOperand(op, 0);
    const TensorType& targets = intOperand(op, 1);
    if (logits.getRank() != 2 && logits.getRank() != 3) {
        throw IRError("cross_entropy logits must be 2-D or 3-D, got " + logits.toString());
    }
    Shape expected(logits.getShape().begin(), logits.getShape().end() - 1);
    if (targets.getShape() != expected) {
        throw IRError("cross_entropy targets " + targets.toString() + " don't match logits " + logits.toString());
    }
    return tensorType(DataType::Float32, {});
}

TypePtr inferLayerNorm(const Operation& op) {
    const TensorType& x = floatOperand(op, 0);
    if (x.getRank() == 0) throw IRError("layernorm needs an operand of rank >= 1");
    Shape feature{x.getShape().back()};
    for (size_t idx : {size_t{1}, size_t{2}}) {
        const TensorType& param = floatOperand(op, idx);
        if (param.getShape() != feature) {
            throw IRError(std::string("layernorm ") + (idx == 1 ? "gamma" : "beta") + " must have shape " +
                          shapeString(feature) + ", got " + param.toString());
        }
    }
    if (!(op.getDoubleAttr("eps") > 0.0)) throw IRError("layernorm eps must be positive");
    return tensorType(DataType::Float32, x.getShape());
}

TypePtr inferReduction(const Operation& op) {
    const TensorType& x = floatOperand(op, 0);
    int64_t rank = x.getRank();
    std::vector<bool> reduced(rank, false);
    for (int64_t axis : op.getIntListAttr("axes")) {
        int64_t a = normalizeAxis(op, axis, rank);
        if (reduced[a]) throw IRError("'" + op.getOpName() + "' lists axis " + std::to_string(a) + " twice");
        reduced[a] = true;
    }
    int64_t keepdims = op.getIntAttr("keepdims");
    if (keepdims != 0 && keepdims != 1) throw IRError("keepdims of '" + op.getOpName() + "' must be 0 or 1");

    Shape out;
    for (int64_t i = 0; i < rank; ++i) {
        if (!reduced[i]) out.push_back(x.getShape()[i]);
        else if (keepdims) out.push_back(1);
    }
    return tensorType(DataType::Float32, out);
}

TypePtr inferTranspose(const Operation& op) {
    const TensorType& x = operandType(op, 0);
    const auto& perm = op.getIntListAttr("perm");
    const Shape& shape = x.getShape();
    std::vector<bool> seen(shape.size(), false);
    bool valid = perm.size() == shape.size();
    Shape out;
    for (int64_t p : perm) {
        if (!valid || p < 0 || p >= static_cast<int64_t>(shape.size()) || seen[p]) {
            valid = false;
            break;
        }
        seen[p] = true;
        out.push_back(shape[p]);
    }
    if (!valid) throw IRError("transpose perm " + shapeString(perm) + " is not a permutation of the " +
                              std::to_string(shape.size()) + " axes of " + x.toString());
    return tensorType(x.getElementType(), out);
}

TypePtr inferReshape(const Operation& op) {
    const TensorType& x = operandType(op, 0);
    Shape shape = op.getIntListAttr("shape");
    int64_t count = 1;
    for (int64_t d : shape) {
        if (d < 0) throw IRError("reshape dimensions must be non-negative, got " + shapeString(shape));
        if (__builtin_mul_overflow(count, d, &count)) {
            throw IRError("reshape shape " + shapeString(shape) + " is too large");
        }
    }
    if (count != x.getNumElements()) {
        throw IRError("reshape from " + x.toString() + " to " + shapeString(shape) + " changes the element count");
    }
    return tensorType(x.getElementType(), shape);
}

TypePtr inferEmbedding(const Operation& op) {
    const TensorType& table = floatOperand(op, 0);
    const TensorType& indices = intOperand(op, 1);
    if (table.getRank() != 2) throw IRError("embedding table must be 2-D, got " + table.toString());
    Shape out = indices.getShape();
    out.push_back(table.getShape()[1]);
    return tensorType(DataType::Float32, out);
}

TypePtr inferConstant(const Operation& op) {
    op.getDoubleAttr("value");
    if (op.getNumResults() != 1 || !op.getResult(0)->getType() || !op.getResult(0)->getType()->isTensor()) {
        throw IRError("'constant' needs an explicit result type");
    }
    return op.getResult(0)->getType();
}

TypePtr inferBroadcastTo(const Operation& op) {
    const TensorType& x = operandType(op, 0);
    Shape shape = op.getIntListAttr("shape");
    Shape out;
    if (!computeBroadcastShape(x.getShape(), shape, out) || out != shape) {
        throw IRError("broadcast_to cannot broadcast " + x.toString() + " to " + shapeString(shape));
    }
    return tensorType(x.getElementType(), shape);
}

TypePtr inferScatterAdd(const Operation& op) {
    const TensorType& indices = intOperand(op, 0);
    const TensorType& updates = floatOperand(op, 1);
    int64_t rows = op.getIntAttr("num_rows");
    if (rows <= 0) throw IRError("scatter_add num_rows must be positive");
    Shape expected = indices.getShape();
    if (updates.getRank() != indices.getRank() + 1 ||
        !std::equal(expected.begin(), expected.end(), updates.getShape().begin())) {
        throw IRError("scatter_add updates " + updates.toString() + " must have the indices' shape " +
                      indices.toString() + " plus one dimension");
    }
    return tensorType(DataType::Float32, {rows, updates.getShape().back()});
}

TypePtr inferCast(const Operation& op) {
    const TensorType& x = operandType(op, 0);
    const std::string& to = op.getStringAttr("to");
    if (to == "f32") return tensorType(DataType::Float32, x.getShape());
    if (to == "i32") return tensorType(DataType::Int32, x.getShape());
    throw IRError("cast 'to' must be \"f32\" or \"i32\", got \"" + to + "\"");
}

TypePtr inferOneHot(const Operation& op) {
    const TensorType& indices = intOperand(op, 0);
    int64_t numClasses = op.getIntAttr("num_classes");
    if (numClasses <= 0) throw IRError("onehot num_classes must be positive");
    Shape out = indices.getShape();
    out.push_back(numClasses);
    return tensorType(DataType::Float32, out);
}

int64_t operandRank(const Operation& op) {
    if (op.getNumOperands() == 0 || !op.getOperand(0) || !op.getOperand(0)->getType() ||
        !op.getOperand(0)->getType()->isTensor()) {
        return -1;
    }
    return static_cast<const TensorType&>(*op.getOperand(0)->getType()).getRank();
}

void defaultAxis(Operation& op) {
    if (!op.hasAttribute("axis")) op.setAttribute("axis", int64_t{-1});
}

void defaultEps(Operation& op) {
    if (!op.hasAttribute("eps")) op.setAttribute("eps", 1e-5);
}

void defaultReduction(Operation& op) {
    int64_t rank = operandRank(op);
    if (!op.hasAttribute("axes") && rank >= 0) {
        std::vector<int64_t> axes(rank);
        std::iota(axes.begin(), axes.end(), 0);
        op.setAttribute("axes", axes);
    }
    if (!op.hasAttribute("keepdims")) op.setAttribute("keepdims", int64_t{0});
}

// Transpose flags are written only when set, so each matmul has one textual form.
void canonicalMatmulFlags(Operation& op) {
    for (const char* name : {"transpose_a", "transpose_b"}) {
        if (op.hasAttribute(name) && attrHasKind(op.getAttribute(name), AttrKind::Int) && op.getIntAttr(name) == 0) {
            op.removeAttribute(name);
        }
    }
}

void defaultPerm(Operation& op) {
    int64_t rank = operandRank(op);
    if (!op.hasAttribute("perm") && rank >= 0) {
        std::vector<int64_t> perm(rank);
        std::iota(perm.rbegin(), perm.rend(), 0);
        op.setAttribute("perm", perm);
    }
}

OpDef makeOp(std::string name, int numOperands, OpClass opClass, std::vector<AttrSpec> attrs,
             std::function<void(Operation&)> setDefaults, std::function<TypePtr(const Operation&)> inferType) {
    OpDef def;
    def.name = std::move(name);
    def.numOperands = numOperands;
    def.hasResult = static_cast<bool>(inferType);
    def.opClass = opClass;
    def.attrs = std::move(attrs);
    def.setDefaults = std::move(setDefaults);
    def.inferType = std::move(inferType);
    return def;
}

} // namespace

const AttrSpec* OpDef::findAttr(std::string_view attrName) const {
    for (const auto& spec : attrs) {
        if (spec.name == attrName) return &spec;
    }
    return nullptr;
}

OpRegistry::OpRegistry() {
    for (const char* name : {"add", "sub", "mul", "div"}) {
        ops_.push_back(makeOp(name, 2, OpClass::Elementwise, {}, nullptr, inferBinary));
    }
    for (const char* name : {"relu", "gelu", "exp", "log"}) {
        ops_.push_back(makeOp(name, 1, OpClass::Elementwise, {}, nullptr, inferUnary));
    }
    ops_.push_back(makeOp("matmul", 2, OpClass::Matmul,
                          {{"transpose_a", AttrKind::Int, false}, {"transpose_b", AttrKind::Int, false}},
                          canonicalMatmulFlags, inferMatmul));
    for (const char* name : {"softmax", "log_softmax"}) {
        ops_.push_back(makeOp(name, 1, OpClass::RowReduction, {{"axis", AttrKind::Int, false}}, defaultAxis,
                              inferAxisNormalization));
    }
    ops_.push_back(makeOp("cross_entropy", 2, OpClass::Loss, {}, nullptr, inferCrossEntropy));
    ops_.push_back(makeOp("layernorm", 3, OpClass::RowReduction, {{"eps", AttrKind::Double, false}}, defaultEps,
                          inferLayerNorm));
    for (const char* name : {"sum", "mean"}) {
        ops_.push_back(makeOp(name, 1, OpClass::Reduction,
                              {{"axes", AttrKind::IntList, false}, {"keepdims", AttrKind::Int, false}},
                              defaultReduction, inferReduction));
    }
    ops_.push_back(makeOp("transpose", 1, OpClass::Shape, {{"perm", AttrKind::IntList, false}}, defaultPerm,
                          inferTranspose));
    ops_.push_back(makeOp("reshape", 1, OpClass::Shape, {{"shape", AttrKind::IntList, true}}, nullptr, inferReshape));
    ops_.push_back(makeOp("embedding", 2, OpClass::Gather, {}, nullptr, inferEmbedding));
    ops_.push_back(makeOp("constant", 0, OpClass::Constant, {{"value", AttrKind::Double, true}}, nullptr,
                          inferConstant));
    ops_.push_back(makeOp("onehot", 1, OpClass::Gather, {{"num_classes", AttrKind::Int, true}}, nullptr,
                          inferOneHot));
    // Backward primitives.
    for (const char* name : {"tanh", "sqrt", "rsqrt"}) {
        ops_.push_back(makeOp(name, 1, OpClass::Elementwise, {}, nullptr, inferUnary));
    }
    ops_.push_back(makeOp("cmp_gt", 2, OpClass::Elementwise, {}, nullptr, inferBroadcastAll));
    ops_.push_back(makeOp("select", 3, OpClass::Elementwise, {}, nullptr, inferBroadcastAll));
    ops_.push_back(makeOp("broadcast_to", 1, OpClass::Shape, {{"shape", AttrKind::IntList, true}}, nullptr,
                          inferBroadcastTo));
    ops_.push_back(makeOp("reduce_max", 1, OpClass::Reduction,
                          {{"axes", AttrKind::IntList, false}, {"keepdims", AttrKind::Int, false}},
                          defaultReduction, inferReduction));
    ops_.push_back(makeOp("scatter_add", 2, OpClass::Scatter, {{"num_rows", AttrKind::Int, true}}, nullptr,
                          inferScatterAdd));
    ops_.push_back(makeOp("cast", 1, OpClass::Elementwise, {{"to", AttrKind::String, true}}, nullptr, inferCast));
    ops_.push_back(makeOp("return", -1, OpClass::Terminator, {}, nullptr, nullptr));

    for (size_t i = 0; i < ops_.size(); ++i) index_[ops_[i].name] = i;
}

const OpRegistry& OpRegistry::instance() {
    static const OpRegistry registry;
    return registry;
}

const OpDef* OpRegistry::lookup(std::string_view name) const {
    auto it = index_.find(std::string(name));
    return it == index_.end() ? nullptr : &ops_[it->second];
}

std::string opClassName(OpClass opClass) {
    switch (opClass) {
        case OpClass::Elementwise: return "elementwise";
        case OpClass::Matmul: return "matmul";
        case OpClass::Reduction: return "reduction";
        case OpClass::RowReduction: return "row-reduction";
        case OpClass::Loss: return "loss";
        case OpClass::Shape: return "shape";
        case OpClass::Gather: return "gather";
        case OpClass::Scatter: return "scatter";
        case OpClass::Constant: return "constant";
        case OpClass::Terminator: return "terminator";
    }
    return "unknown";
}

std::string attrKindName(AttrKind kind) {
    switch (kind) {
        case AttrKind::Int: return "int";
        case AttrKind::Double: return "float";
        case AttrKind::IntList: return "int-list";
        case AttrKind::String: return "string";
    }
    return "unknown";
}

bool attrHasKind(const AttributeValue& value, AttrKind kind) {
    switch (kind) {
        case AttrKind::Int: return std::holds_alternative<int64_t>(value);
        case AttrKind::Double: return std::holds_alternative<double>(value);
        case AttrKind::IntList: return std::holds_alternative<std::vector<int64_t>>(value);
        case AttrKind::String: return std::holds_alternative<std::string>(value);
    }
    return false;
}

void finalizeOp(Operation& op, bool inferResultType) {
    const OpDef* def = OpRegistry::instance().lookup(op.getOpName());
    if (!def) throw IRError("unknown op '" + op.getOpName() + "'");
    if (def->setDefaults) def->setDefaults(op);
    if (inferResultType && def->hasResult) op.getResult(0)->setType(def->inferType(op));
}

} // namespace dlc
