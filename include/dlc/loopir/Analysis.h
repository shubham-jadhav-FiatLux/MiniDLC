#pragma once

#include "dlc/loopir/LoopIR.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace dlc::loopir {

// A closed integer interval; the sides saturate at the int64 limits when unknown.
struct Interval {
    int64_t lo;
    int64_t hi;
    static Interval unknown();
    bool known() const;
    bool within(int64_t first, int64_t last) const { return lo >= first && hi <= last; }
};

// Integer facts in scope: each loop variable spans [0, extent - 1]; an integer local spans its value's range.
class RangeEnv {
public:
    void push() { scopes_.emplace_back(); }
    void pop() { scopes_.pop_back(); }
    void set(const std::string& name, Interval range) { scopes_.back()[name] = range; }
    std::optional<Interval> get(const std::string& name) const;

private:
    std::vector<std::map<std::string, Interval>> scopes_{1};
};

// A conservative interval of an integer expression. Values loaded from memory are unknown.
Interval rangeOf(const ExprPtr& expr, const RangeEnv& env);

// Σ coeffs[var]·var + constant, when the expression is affine in its variables (div/mod/loads make it non-affine).
struct Affine {
    std::map<std::string, int64_t> coeffs;
    int64_t constant{0};
};
std::optional<Affine> affineForm(const ExprPtr& expr);

struct BoundsReport {
    int accesses{0};     // loads and stores checked
    int proven{0};       // proven in bounds by range analysis
    int guarded{0};      // indexed by data: given a guard
    int eliminated{0};   // guards that range analysis proved true and removed
};

// Guard insertion + bounds-check elimination. Every load and store is checked against its buffer with range
// analysis. An access indexed by data (an index loaded from memory: embedding ids, class labels, scatter rows) gets a
// guard — a load becomes select(in bounds, load(clamped index), 0), a store runs only if its index is in bounds — so
// the code is memory-safe for any input. Any other access must be proven in bounds (else IRError: a lowering bug).
// Then every `if` whose condition range analysis proves always true (or false) is replaced by the branch it takes.
BoundsReport checkBounds(Kernel& kernel);
// The ablation of the above: every access is checked at run time (loads read 0 and stores are skipped out of
// bounds), nothing is proven or eliminated.
BoundsReport checkEveryAccess(Kernel& kernel);

// Dependence check of the schedule (IRError if it fails). Iterations of a Parallel loop must write disjoint elements of
// every buffer, and read only elements that no other iteration writes; this enumerates the loop's iterations, with
// range analysis over the inner loops. A Vectorized loop must not carry a dependence through global memory: each store
// varies with the vector variable, and a buffer both read and written in the loop is read only at the element written.
// (Register buffers indexed independently of the vector variable are reductions, which the loop may reassociate.)
void verifySchedule(const Kernel& kernel);

} // namespace dlc::loopir
