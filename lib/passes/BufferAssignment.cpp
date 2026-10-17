#include "dlc/passes/BufferAssignment.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/SlotTracker.h"
#include "dlc/passes/CostModel.h"
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <unordered_set>

namespace dlc {

std::string regionName(Region region) { return region == Region::Persistent ? "persistent" : "transient"; }

namespace {

int64_t align(int64_t n) {
    return (n + BufferAssignment::kAlignment - 1) / BufferAssignment::kAlignment * BufferAssignment::kAlignment;
}

bool liveTogether(const Buffer& a, const Buffer& b) { return a.begin <= b.end && b.begin <= a.end; }

bool bytesOverlap(const Buffer& a, const Buffer& b) {
    return a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}

// A kernel may overwrite `arg` with one of its outputs if it reads `arg` only at the element it writes: an
// elementwise kernel over arg's shape, or a matmul whose epilogue (not the matmul itself) reads it.
bool writesInPlaceSafely(const Kernel& writer, const Value* arg) {
    if (static_cast<const TensorType&>(*arg->getType()).getShape() != writer.iterationShape) return false;
    if (writer.kind == KernelKind::Elementwise) return true;
    if (writer.kind != KernelKind::Matmul) return false;
    for (const Operation* op : writer.ops) {
        if (op->getOpName() != "matmul") continue;
        for (const auto& operand : op->getOperands()) {
            if (operand.get() == arg) return false;
        }
    }
    return true;
}

} // namespace

BufferAssignment::BufferAssignment(const Function& func, const std::vector<Kernel>& kernels)
    : func_(func), kernels_(kernels) {
    const int64_t K = static_cast<int64_t>(kernels.size());
    const auto& args = func.getArguments();
    const Operation* ret = func.getEntryBlock()->getOperations().back().get();

    std::unordered_map<const Operation*, int64_t> kernelOfOp;
    for (int64_t k = 0; k < K; ++k) {
        for (const Operation* op : kernels[k].ops) kernelOfOp[op] = k;
        for (const Value* out : kernels[k].outputs) producer_[out] = k;
    }
    // Reshapes alias their operand.
    for (const auto& op : func.getEntryBlock()->getOperations()) {
        if (op->getOpName() == "reshape") aliasOf_[op->getResult(0).get()] = op->getOperand(0).get();
    }

    // In-place updates: a donated result writes into its argument's buffer if the kernel that computes it reads the
    // argument (if at all) only at the element it writes, and runs after every other reader.
    std::unordered_map<const Value*, std::vector<int64_t>> readers; // root value → kernels that read it
    for (int64_t k = 0; k < K; ++k) {
        for (const Value* in : kernels[k].inputs) readers[root(in)].push_back(k);
    }
    std::unordered_set<const Value*> returnedPlain;
    const auto& info = func.getResultInfo();
    for (size_t i = 0; i < ret->getNumOperands(); ++i) {
        if (i >= info.size() || info[i].kind != ResultKind::Donates) returnedPlain.insert(root(ret->getOperand(i).get()));
    }
    std::unordered_set<const Value*> donatedValues;
    for (size_t i = 0; i < ret->getNumOperands() && i < info.size(); ++i) {
        if (info[i].kind != ResultKind::Donates) continue;
        const Value* value = ret->getOperand(i).get();
        const Value* arg = args.at(info[i].arg).get();
        if (value == arg) continue; // unchanged: nothing to write
        auto prod = producer_.find(value);
        bool safe = prod != producer_.end() && !aliasOf_.count(value) && donatedValues.insert(value).second &&
                    !returnedPlain.count(arg);
        if (safe) {
            safe = writesInPlaceSafely(kernels[prod->second], arg);
            for (int64_t k : readers[arg]) safe = safe && k <= prod->second;
        }
        if (safe) {
            inPlace_[value] = arg;
            aliasOf_[value] = arg;
        } else {
            finalCopies_.push_back({value, arg});
        }
    }

    // Live ranges of the buffer roots.
    std::unordered_map<const Value*, std::pair<int64_t, int64_t>> range;
    auto touch = [&](const Value* v, int64_t k) {
        const Value* r = root(v);
        auto [it, inserted] = range.try_emplace(r, k, k);
        if (!inserted) {
            it->second.first = std::min(it->second.first, k);
            it->second.second = std::max(it->second.second, k);
        }
    };
    for (const auto& arg : args) touch(arg.get(), -1);
    for (int64_t k = 0; k < K; ++k) {
        for (const Value* out : kernels[k].outputs) touch(out, k);
        for (const Value* in : kernels[k].inputs) touch(in, k);
    }
    for (const auto& v : ret->getOperands()) touch(v.get(), K);
    for (const auto& copy : finalCopies_) touch(copy.from, K);

    // Persistent region: params, state and consts, live for the whole session.
    for (size_t i = 0; i < args.size(); ++i) {
        ArgRole role = func.getArgRole(i);
        if (role == ArgRole::Input) continue;
        Buffer b{args[i].get(), Region::Persistent, persistentSize_, tensorBytes(*args[i]), -1, K};
        persistentSize_ += align(b.size);
        bufferIndex_[b.value] = buffers_.size();
        buffers_.push_back(b);
    }
    // Transient arena: inputs and kernel outputs, placed largest first at the lowest offset that is free for their
    // whole live range.
    std::vector<Buffer> transient;
    for (const auto& [value, live] : range) {
        if (bufferIndex_.count(value)) continue;
        bool isArg = !value->getDefiningOp();
        if (!isArg && !producer_.count(value)) continue; // an inlined constant: no buffer
        transient.push_back({value, Region::Transient, 0, tensorBytes(*value), live.first, live.second});
    }
    std::sort(transient.begin(), transient.end(), [](const Buffer& a, const Buffer& b) {
        if (a.size != b.size) return a.size > b.size;
        if (a.begin != b.begin) return a.begin < b.begin;
        return a.value->getId() < b.value->getId();
    });
    std::vector<Buffer> placed;
    for (Buffer b : transient) {
        transientUnshared_ += align(b.size);
        std::vector<const Buffer*> conflicts;
        for (const Buffer& p : placed) {
            if (liveTogether(p, b)) conflicts.push_back(&p);
        }
        std::sort(conflicts.begin(), conflicts.end(), [](const Buffer* x, const Buffer* y) { return x->offset < y->offset; });
        int64_t offset = 0;
        for (const Buffer* c : conflicts) {
            if (offset + b.size <= c->offset) break;
            offset = std::max(offset, align(c->offset + c->size));
        }
        b.offset = offset;
        transientSize_ = std::max(transientSize_, align(offset + b.size));
        placed.push_back(b);
    }
    for (const Buffer& b : placed) {
        bufferIndex_[b.value] = buffers_.size();
        buffers_.push_back(b);
    }
}

const Value* BufferAssignment::root(const Value* value) const {
    for (auto it = aliasOf_.find(value); it != aliasOf_.end(); it = aliasOf_.find(value)) value = it->second;
    return value;
}

bool BufferAssignment::hasBuffer(const Value* value) const { return bufferIndex_.count(root(value)) != 0; }

const Buffer& BufferAssignment::bufferOf(const Value* value) const {
    auto it = bufferIndex_.find(root(value));
    if (it == bufferIndex_.end()) throw IRError("buffer assignment: a value has no buffer");
    return buffers_[it->second];
}

bool BufferAssignment::updatesInPlace(const Value* value) const { return inPlace_.count(value) != 0; }

void BufferAssignment::verify() const {
    SlotTracker names(func_);
    for (size_t i = 0; i < buffers_.size(); ++i) {
        const Buffer& a = buffers_[i];
        if (a.offset % kAlignment != 0) throw IRError("buffer assignment: %" + names.name(a.value) + " is misaligned");
        for (size_t j = i + 1; j < buffers_.size(); ++j) {
            const Buffer& b = buffers_[j];
            if (a.region == b.region && liveTogether(a, b) && bytesOverlap(a, b)) {
                throw IRError("buffer assignment: %" + names.name(a.value) + " and %" + names.name(b.value) +
                              " are live at the same time and overlap in the " + regionName(a.region) + " region");
            }
        }
    }
    // In-place updates: the writer reads the argument only at the element it writes, after every other reader.
    for (const auto& [value, arg] : inPlace_) {
        int64_t writer = producer_.at(value);
        const Kernel& k = kernels_[writer];
        if (!writesInPlaceSafely(k, arg)) {
            throw IRError("buffer assignment: " + k.name + " cannot update %" + names.name(arg) + " in place");
        }
        for (size_t kk = 0; kk < kernels_.size(); ++kk) {
            for (const Value* in : kernels_[kk].inputs) {
                if (root(in) == arg && static_cast<int64_t>(kk) > writer) {
                    throw IRError("buffer assignment: %" + names.name(arg) + " is read by " + kernels_[kk].name +
                                  " after " + k.name + " overwrote it");
                }
            }
        }
    }
}

std::string BufferAssignment::report(const Function& func) const {
    SlotTracker names(func);
    std::ostringstream os;
    char line[256];
    os << "Buffer assignment for @" << func.getName() << " (" << kernels_.size() << " kernels, "
       << kAlignment << "-byte alignment)\n";
    std::snprintf(line, sizeof line, "  persistent region: %lld bytes (params, optimizer state, consts)\n",
                  static_cast<long long>(persistentSize_));
    os << line;
    std::snprintf(line, sizeof line, "  transient arena:   %lld bytes, vs %lld without reuse (%.1f%% less)\n",
                  static_cast<long long>(transientSize_), static_cast<long long>(transientUnshared_),
                  transientUnshared_ == 0 ? 0.0
                                          : 100.0 * (1.0 - static_cast<double>(transientSize_) / transientUnshared_));
    os << line;
    std::snprintf(line, sizeof line, "  in-place updates:  %zu, copied at the end: %zu\n", inPlace_.size(),
                  finalCopies_.size());
    os << line;
    std::vector<const Buffer*> sorted;
    for (const Buffer& b : buffers_) sorted.push_back(&b);
    std::sort(sorted.begin(), sorted.end(), [](const Buffer* a, const Buffer* b) {
        if (a->region != b->region) return a->region < b->region;
        return a->offset != b->offset ? a->offset < b->offset : a->value->getId() < b->value->getId();
    });
    for (const Buffer* b : sorted) {
        std::snprintf(line, sizeof line, "  %%%-28s %-10s offset %10lld  size %10lld  live %lld..%lld\n",
                      names.name(b->value).c_str(), regionName(b->region).c_str(), static_cast<long long>(b->offset),
                      static_cast<long long>(b->size), static_cast<long long>(b->begin),
                      static_cast<long long>(b->end));
        os << line;
    }
    return os.str();
}

bool BufferAssignmentPass::runOnFunction(Function& func) {
    FusionPass fusion(/*annotate=*/false);
    fusion.runOnFunction(func);
    BufferAssignment assignment(func, fusion.getKernels());
    assignment.verify();
    report_ = assignment.report(func);
    if (!annotate_) return false;
    for (const Kernel& k : fusion.getKernels()) {
        for (const Value* out : k.outputs) {
            const Buffer& b = assignment.bufferOf(out);
            Operation* def = out->getDefiningOp();
            def->setAttribute("mem.region", regionName(b.region));
            def->setAttribute("mem.offset", b.offset);
            def->setAttribute("mem.size", b.size);
        }
    }
    return true;
}

} // namespace dlc
