# CS6843 — Program Analysis (IIT Madras, Jul–Nov 2025) — Learning Notes

> Instructor: **Prof. Rupesh Nasre** (IIT Madras). Course page: `~rupesh/teaching/pa/jul25/`. G slot, room CS26.
> This single file holds:
> - lesson-by-lesson notes for all seven topic decks (DFA, PTR, SHA, PAR, SLI, SEC, DYN), with my handwritten slide
>   annotations and the extra reading material folded in;
> - worked answers to the past-paper questions I collected in `PreveousYearsPapers/QuestionForEndsem.txt`. The
>   pointer-analysis answers were checked with small solvers;
> - a review of my four LLVM assignments. Each was **built and executed** against the public, private and reference
>   tests (Oct 2026, using an LLVM 23 toolchain; recipe in §10.6);
> - a Phase-2 fix plan for each assignment;
> - bridges to AI infrastructure and references.
>
> **Evaluation:** four LLVM assignments, 60% in total (5 + 15 + 20 + 20); MidSem 20; EndSem 20. Exams are open-book
> with handwritten notes allowed (no electronics or printouts). MidSem and EndSem cover mutually exclusive topics.
> **Schedule:** Aug DFA (A1) · Sep PTA, SHA (A2, MidSem) · Oct DYN, PAR (A3) · Nov SLI, SEC (A4, EndSem).

---

## TL;DR — status of my work (verified Oct 2026)

| Assignment | What it is | Verified result | Verdict |
|---|---|---|---|
| A1 (5) | LLVM install + IR, CFG dot, call graph, loop info | Report complete and correct | OK |
| A2 (15) | Flow-sensitive intraprocedural **range analysis** (Kildall + widening) | **26/30**: public 15/15, private **11/15** (fails 17, 18, 19, 21) | Missing: calls, `select`, `switch` refinement |
| A3 (20) | **Andersen** points-to + alias query | Submitted (Oct 2): **11/20** (public 10/10, hidden **1/10**). My Dec 2025 rewrite: **20/20** | Submission mishandles globals and `*p = &x`; fixed later |
| A4 (20) | **Array bounds-check instrumentation** driven by range analysis | Public **10/10**. My 5 hidden-style tests: **0/5** | No back-edge propagation, no path infeasibility, wrong array model, unsound join refinement |

The TA reference solutions are on disk and passed the same harnesses: `Assignment_2/template/Template/rangeAnalysis.cc`
(30/30) and `Assignment_3/template/assignment3/testcases_sol/sol.cc` (20/20). The harnesses are therefore trustworthy.

---

## Table of contents

1. Foundations: what program analysis is, compilers, LLVM IR for this course
2. Data-flow analysis (DFA): RD, LV, AE, VBE, lattices, monotone/distributive frameworks, MFP/MOP, constant
   propagation, range analysis and widening
3. Analysis dimensions, abstract interpretation, soundness/completeness
4. Pointer analysis (PTR): Andersen, Steensgaard, constraint graphs, cycle/dominator optimizations, NP-hardness of
   ordering, prioritized PTA, parallel replication-based PTA, graph rewrite rules
5. Shape analysis (SHA): Ghiya–Hendren D/I matrices and the exact inference rules
6. Parallelization (PAR): dependences, ILP and GCD tests, races, locks/atomics/barriers, irregular programs
7. Program slicing (SLI): Weiser's equations, PDG, static/dynamic slicing
8. Security analysis (SEC): buffer overruns, annotations, polyhedral bounds checks, tools, Hoare logic
9. Dynamic analysis (DYN): profiling, edge profiling, Ball–Larus path profiling
10. My assignments: design, verified results, root causes, Phase-2 fixes, how to re-run everything
11. Revision Q&A (past papers, with worked answers)
12. Bridges to AI infrastructure
13. References

---

## 1. Foundations

### 1.1 What is program analysis?

*Definition (slide 2):* for an **end goal**, identify **"interesting aspects"** of a **program representation**
(source, AST, IR, binary, or the trace of executed instructions).

- **Static analysis** runs at compile time over the code. It covers all inputs, so it must over-approximate.
- **Dynamic analysis** observes an execution. It is precise for that input and says nothing about other inputs.

| End goal | Interesting aspect |
|---|---|
| Dead-code elimination | Reachability |
| Constant propagation | use–def chains |
| Security | Array index ranges, dangling pointers |
| Parallelization | Data dependence, SIMD opportunities |
| Debugging | Slices |
| Cache performance | Memory access pattern |
| Memory reduction, register allocation | Live ranges |

*Analysis is a precursor to optimization; transformation is the post-cursor.* **Optimization = analysis +
transformation.**

### 1.2 Three motivating examples (slides 8–10)

1. **CSE needs pointer analysis.** In `p = &a; c = a + b; d = *p + b;`, `*p + b` is `a + b` only if we know that
   `p → {a}` and nothing between the two statements writes `a` or `b`. After that, `t = a + b; c = t; d = t;`. Typing
   the temporary `t` also needs type analysis (the types of `a` and `b`).
2. **Parallelizing two functions needs dependence and race analysis.** `T1: z = x + 1;` and `T2: if (flag) y = x - 1;`
   can run in parallel. A **data race** needs (1) at least two threads, (2) a common memory location (possibly
   through aliases), and (3) at least one write.
3. **Security needs taint analysis.** If the input is compromised, can `fun` return a compromised string? This is
   reachability of input to `y`.

### 1.3 Compiler organization

Source → lexer (tokens) → parser (syntax tree) → semantic analyzer → intermediate code generator (IR: 3-address code,
quadruples, triples, DAG) → **machine-independent optimizer** → code generator → machine-dependent optimizer → target
code, with a symbol table shared across stages. Analyses can run on source, AST, IR or machine code; the course uses
**LLVM IR**.

### 1.4 LLVM IR primer (what the assignments relied on)

Everything below is clang `-O0 -g` output, which is what every test used.

- **Locals live in memory.** Each local gets `%x = alloca i32`. Reads are `load`, writes are `store`. Arguments are
  stored into allocas in the entry block (`store i32 %0, ptr %3`). SSA values (`%5 = add nsw i32 %3, %4`) are
  temporaries, so analyses track **both** allocas (program variables) and SSA registers.
- **Names come from debug records.** `#dbg_declare(ptr %3, !22, ...)` maps an alloca to the variable `arg1`. The
  pattern is `for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) name = DVR.getVariable()->getName();`
  (A2 and A3 printing).
- **Array access** is `%idx64 = sext i32 %i to i64` followed by
  `%p = getelementptr inbounds [10 x i32], ptr %arr, i64 0, i64 %idx64`, then `load` or `store` through `%p`.
  **A GEP never touches memory.** It is pure address arithmetic: `%idx2 = getelementptr i32, ptr %MyVar, i64 1`
  performs **0 loads** (a past MidSem question). Array initialisation `= {0}` becomes `call @llvm.memset`.
- **Control flow.** `icmp slt/sle/sgt/sge`, then `br i1 %c, label %T, label %F`. `switch i32 %v, label %default
  [i32 10, label %a ...]` for `switch`. `select i1 %c, i32 30, i32 40` for a simple `?:`. `phi` nodes are rare at -O0.
- **Calls.** `call i32 @llvm.abs.i32(...)`: clang turns a user-written `abs` into the intrinsic. Passing `&a` to
  `printf`/`scanf` means `a` may change.
- **New pass manager plugin:**
  ```cpp
  extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "Name", "v0.1", [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback([](StringRef N, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
        if (N == "range-analysis") { FPM.addPass(MyPass()); return true; } return false; }); }};
  }
  // run:   opt -load-pass-plugin ./a2Pass.so -passes='range-analysis' -disable-output t.ll
  ```
  (In LLVM 23 the header moved from `llvm/Passes/PassPlugin.h` to `llvm/Plugins/PassPlugin.h`.)
- **Useful APIs:**
  - `llvm::FindFunctionBackedges(F, SmallVector<pair<const BB*,const BB*>>&)` finds loop back edges (A2).
  - `isPotentiallyReachable(A, B)` answers CFG reachability.
  - `BB->splitBasicBlock(I)` and `IRBuilder<>` with `CreateICmpSGE/SLT`, `CreateOr` / `CreateLogicalOr` (which emits
    `select i1 a, i1 true, i1 b`), `CreateCondBr` and `CreateRet` are what A4 instrumentation uses.
  - `successors(BB)` and `predecessors(BB)` walk the CFG.
- **A1 commands:**
  - `clang -S -emit-llvm x.c -o x.ll`
  - `opt -passes=dot-cfg x.ll` (writes `.main.dot` etc.; view with `xdot`)
  - `opt -passes=print-callgraph x.ll`
  - `opt -passes='print<loops>' x.ll` (prints e.g. `Loop at depth 1 containing: %2<header><exiting>,%5,%9<latch>`)
- **Testing:** lit + FileCheck. A `; RUN:` line pipes `opt` stderr into `FileCheck %s`. `; CHECK:` lines must match
  **in order** (A2 prints variables in sorted name order, which is why the reference uses `std::map<StringRef,...>`).

---

## 2. Data-flow analysis (DFA)

### 2.1 CFG and basic blocks

- A **basic block** is a maximal straight-line sequence with one entry and one exit.
  - **Leaders** are: the first instruction, any target of a jump, and any instruction right after a jump.
- **CFG:** nodes are blocks; edges are possible control transfers. Classwork (Collatz-like loop):
  `START → (x=0; n=0; scanf) → [x != 1] → [x % 2] → {x = 3x+1; ++n} | {x /= 2; n = n+1} → back to [x != 1] → printf → END`.
- **Optimization chain example (`gcc -O2`).** `int x = 1; if (x > 0) ++x; else x = 100; printf(x)` becomes
  `if (1 > 0)` by constant propagation, the `else` disappears by dead-code elimination, and finally
  `printf("%d\n", 2)` by constant folding.

### 2.2 Reaching definitions (RD)

- **Definition:** a definition *d* **reaches** point *p* if some path from just after *d* to *p* has no
  redefinition (**kill**) of *d*'s variable.
- **Equations (forward, may):**
  - `in(B) = ∪ out(P)` over predecessors `P`
  - `out(B) = gen(B) ∪ (in(B) − kill(B))`
  - initially `out(B) = {}`
  - gen(B) = definitions in B that reach B's end; kill(B) = all other definitions of the variables B defines.
- **Algorithm (round-robin):**
  ```
  for each B: compute gen(B), kill(B); out(B) = {}
  do { for each B: in(B) = ∪ out(P); out(B) = gen(B) ∪ (in(B) − kill(B)) } while any in(B) changed
  ```
  "Can you do better?" Yes: use a **worklist** (only revisit successors of changed blocks) and visit blocks in
  **reverse postorder** for forward problems (postorder for backward ones).
- **Worked example (slides).**
  - Blocks: `B0 = {D0: y=3; D1: x=10; D2: y=11; if c}`, `B1 = {D3: x=1; D4: y=2}`, `B2 = {D5: z=x; D6: x=4}`, and B3
    is the join.
  - gen: B0 {D1,D2}, B1 {D3,D4}, B2 {D5,D6}, B3 {}.
  - kill: B0 {D3,D4,D6}, B1 {D0,D1,D2,D6}, B2 {D1,D3}.
  - Fixpoint: out(B0)={D1,D2}, out(B1)={D3,D4}, out(B2)={D2,D5,D6}, so **in(B3) = {D2,D3,D4,D5,D6}**.
  - D2 survives through B2 because B2 redefines only x and z.
- **Classwork with a loop (recomputed by script):**
  - CFG: B0→B1, B0→B2, B1→B3, B2→B3, B3→B0.
  - gen: B0 {D1,D2}, B1 {D3,D4}, B2 {D5,D6}, B3 {D7,D8}.
  - kill: B0 {D3,D4,D6,D8}, B1 {D1,D2,D6,D8}, B2 {D2,D3,D7,D8}, B3 {D2,D3,D5,D6}.
  - Fixpoint after 2 passes (stable on the 3rd):

    | Block | in | out |
    |---|---|---|
    | B0 | {D1,D4,D7,D8} | {D1,D2,D7} |
    | B1 | {D1,D2,D7} | {D3,D4,D7} |
    | B2 | {D1,D2,D7} | {D1,D5,D6} |
    | B3 | {D1,D3,D4,D5,D6,D7} | {D1,D4,D7,D8} |
- **RD framework summary:**
  - Domain: sets of definitions.
  - Direction: forward.
  - Meet: ∪.
  - Initialization: out = {}.
  - Uses: use–def chains, constant propagation, loop-invariant detection.

### 2.3 Live variables (LV)

- **Definition:** *v* is **live** at *p* if *v* is used along some path starting at *p* before being redefined;
  otherwise *v* is dead.
- **Motivation (memory/register reuse).** In `int x=2,y=3,z=1; if (x==2){ y=z; x=9; y=7; x=x-y; } else { y=x+z; ++x; } printf(y)`:
  - y is dead at lines 2, 3 and 4 (its value is overwritten before any use) and in the else block before line 7.
  - So y and z can share a register or memory slot.
  - This is the core of **register allocation**: build an interference graph of live ranges, then colour it.
- **Equations (backward, may):**
  - `out(B) = ∪ in(S)` over successors `S`
  - `in(B) = use(B) ∪ (out(B) − def(B))`
  - initially `in(B) = {}`
  - use(B) = variables read before any write in B; def(B) = variables written in B.

### 2.4 Available expressions and very busy expressions

| | Forward | Backward |
|---|---|---|
| **∪ (may)** | Reaching definitions | Live variables |
| **∩ (must)** | Available expressions | Very busy expressions |

- **Available at P:** the expression is computed on **every** path from START to P and not invalidated afterwards
  (no operand redefined). Used for **CSE**.
- **Very busy at P:** on **every** path from P to END the expression is computed before any operand changes. Used
  for **code hoisting**.
- Must-analyses initialise non-entry blocks to the **universal set** (⊤ for ∩), otherwise loops yield nothing.

### 2.5 The data-flow framework (theory)

A framework is **(D, V, ∧, F)**: direction, value domain, meet (confluence) operator, and transfer functions, plus
boundary and initial values. A *point* is the start or end of a basic block.

**Semilattice (V, ⊓):**
- ⊓ is idempotent (a ⊓ a = a), commutative and associative.
- It induces a partial order: `a ≥ b ⇔ a ⊓ b = b`. Here "≥" means "more optimistic / more precise".
- **⊤** satisfies a ⊓ ⊤ = a; **⊥** satisfies a ⊓ ⊥ = ⊥.
- For RD (meet = ∪), ⊤ = {} and ⊥ = the set of all definitions. The slide draws the powerset lattice "upside down":
  {} on top, {d1,d2,d3} at the bottom.
- **Height** is the longest chain. For bit-vector problems it is the number of bits, which is finite, so iteration
  terminates.

**Monotone framework:**
- Definition: `x ≥ y ⇒ f(x) ≥ f(y)`, equivalently `f(x ⊓ y) ≤ f(x) ⊓ f(y)`.
- Monotone transfer functions plus a finite-height lattice guarantee **termination**: values only move down and
  can drop at most *height* times per point.

**Distributive framework:**
- Definition: `f(x ⊓ y) = f(x) ⊓ f(y)`. Distributive implies monotone.
- Gen/kill (bit-vector) problems are distributive because `f(x) = gen ∪ (x − kill)`.

**MOP vs MFP:**
- **MOP** (meet over all paths): `MOP(n) = ⊓_{paths K to n} f_K(boundary)`, where f_K composes the block transfer
  functions along path K. This is the best a path-insensitive analysis can do. It is undecidable or exponential in
  general (paths are unbounded with loops; some are infeasible).
- **MFP** (maximal fixed point) is what the iterative algorithm computes when started from ⊤. It is **unique and
  independent of visit order**.
- **MFP ≤ MOP ≤ IDEAL (perfect, feasible paths only).** If the framework is distributive, **MFP = MOP**.
- Intuition: MFP meets at every join and then applies f; MOP applies f along each path and meets at the end. With
  only monotonicity, f(x ⊓ y) can be strictly below f(x) ⊓ f(y).

**Constant propagation is monotone but not distributive** (Lam, CS243 L4 in ExtraMaterial):
- Per-variable lattice: UNDEF (⊤) > {…, −1, 0, 1, …} > NAC (⊥, "not a constant").
- Meet: `c ⊓ c = c`, `c1 ⊓ c2 = NAC`, `UNDEF ⊓ c = c`.
- Counterexample. Path 1 sets `a=2; b=3`, path 2 sets `a=3; b=2`, and the join computes `c = a + b`.
  - MOP: c = 5 on both paths, so c = 5.
  - MFP: at the join a = NAC and b = NAC, so c = NAC.

**Efficiency (Lam L4; Kam–Ullman):**
- With depth-first (reverse postorder) visiting, bit-vector problems converge in **d + 2** passes, where d is the
  maximum number of back edges on any acyclic path (loop nesting depth, ≈ 2–3 in practice).
- A worklist algorithm (**Kildall 1973**) avoids recomputing unchanged blocks.

### 2.6 Range (interval) analysis — the A2 problem

- **Lattice per variable.** Intervals [l, h] with l ≤ h, plus **EMPTY** (bottom, "no value / unreachable") and
  **FULL** = [INT_MIN, INT_MAX] (top in the "conservative" sense).
  - **Merge (join)** is the hull: `[a,b] ⊔ [c,d] = [min(a,c), max(b,d)]`; EMPTY is its identity and FULL absorbs.
  - **Intersect** is used for branch refinement; disjoint intervals give EMPTY.
- **Infinite height** (`[0,0] ⊂ [0,1] ⊂ [0,2] …`) means plain iteration need not terminate, so it needs
  **widening**. The course uses the crude version: at a **back edge**, any variable whose range changed becomes
  FULL. Standard Cousot widening is gentler: only the unstable bound jumps to ±∞, e.g. [0,0], [0,1] → [0, +∞].
  - Example 3 of the spec: `for (i=0; i<10; i++)` gives i FULL at the header and **[10, INT_MAX]** after the loop.
    j stays [10,10] because its range is the same at the start and end of each iteration.
- **Saturating arithmetic.** Compute in 64 bits and clamp to INT_MIN/INT_MAX.
  - `[l1,h1]+[l2,h2] = [l1+l2, h1+h2]`; `−` gives `[l1−h2, h1−l2]`.
  - `*` takes min and max over the four corner products.
  - The reference also special-cases `x * [0,0] = [0,0]` **before** the FULL check, so `FULL * 0 = 0`.
- **Branch refinement belongs on the edge, before the meet.** On the true edge of `x < k`, x becomes
  `x ∩ [INT_MIN, k−1]`; on the false edge, `x ∩ [k, INT_MAX]`. Apply it to the alloca **and** to the loaded SSA
  value.
  - For `switch`, the case edge refines to `[c,c]` (`findCaseDest`); the default edge does not refine.
- **The TA reference** (`Assignment_2/template/Template/rangeAnalysis.cc`) is a clean model to copy:
  - one `BasicBlockState` per block (map Value* → Range);
  - a Kildall worklist;
  - in `meetPredecessorBasicBlockState`, the edge refinement happens inside the meet and **back-edge predecessors
    are met last** with widening to FULL;
  - transfer rules: `alloca` → FULL until stored; `load` copies the alloca's range; `store` writes a constant or a
    tracked value; binary operators; a call returns FULL and makes every alloca passed as an argument FULL; `phi`
    and `select` merge their incoming values;
  - output is printed at the unique exit block.

---

## 3. Analysis dimensions, abstract interpretation, soundness

### 3.1 The four sensitivities (slides 24–30)

"How many hands do you need to tell the time precisely?" More dimensions bring more precision and more cost.

- **Flow-sensitive.** Respects statement order and gives a result per program point (`L0: a=0; L1: a=1;` → a is 0
  after L0 and 1 after L1). Flow-insensitive analysis treats the program as a set of statements: a ∈ {0,1}
  everywhere. It is cheaper, easy to parallelize, and the norm for pointer analysis (Andersen, Steensgaard).
- **Context-sensitive** (interprocedural). Distinguishes call sites: `fun(0)` vs `fun(1)` give y = 0 along L0 and
  y = 1 along L1. Context-insensitive gives y ∈ {0,1}; intraprocedural analysis knows nothing about parameters, so
  y ∈ (−∞, +∞).
  - The number of contexts is **exponential** in call depth (main→f1→g1, main→f1→g2, …), in time and storage.
    Remedies: k-limited call strings, summaries, or merging recursion.
  - "How many ancestor names identify a student?" is the analogy for k-CFA.
- **Path-sensitive.** Tracks branch correlations: `if (a==0) b=1; else b=2;` gives "b is 1 when a is 0". The
  number of paths explodes (c1∧c2∧c3…, nested loops).
- **Field-sensitive.** `s.a = 0; s.b = 1` keeps s.a = 0 and s.b = 1. Field-insensitive analysis collapses
  aggregates (structs, unions, **arrays**) into one variable: s ∈ {0,1}. A4's spec deliberately asked for this:
  *treat the array as a single variable*.

**Classwork (slide 30).** `int g=0; void fun(int n){ g=n; } main(){ int a=1; a=2; fun(a); /*L1*/ a=3; fun(a); /*L2*/ }`

| Analysis | a | n | g |
|---|---|---|---|
| Context- and flow-sensitive | 2 at L1, 3 at L2 | n=2 for call L1, n=3 for call L2 | 2 after L1, 3 after L2 |
| Interprocedural, context-insensitive, flow-sensitive | 2 at L1, 3 at L2 | {2,3} | {2,3} after either call |
| Intraprocedural, flow-insensitive | {1,2,3} | unknown (⊤) | {0} ∪ unknown, i.e. ⊤ |

### 3.2 Concrete vs abstract interpretation

- **Concrete interpretation** runs on actual values. **Abstract interpretation** runs on an approximate abstract
  domain (Cousot & Cousot 1977). It is sound if the abstraction of every concrete result is covered.
- **Example: one bit for "x == 0"**, initialised to F.
  - `x = 0` → T
  - `++x` → F (x ≠ 0)
  - `--x` → **?** The bit cannot say whether x returned to 0, so the answer must be F, the conservative value.
- **Choosing the abstraction is everything.**
  - Bits for "x == 0" and "x < 2", starting at 00: `x=0` → 11, then `++x` → 01.
  - Then `--x` gives **11** if **type information** says x ≥ 0 (x < 2 and x ≠ 0 means x = 1, so --x = 0).
  - Without type information it gives **00**: x could be any value < 2, and wrap-around at INT_MIN breaks even
    "x < 2".
- **Saturating counters.** k = log₂ n bits encode n states. Two are reserved (⊥ = unknown, ⊤ = saturated or
  overflow), so **n − 2 values are stored faithfully**. Extra knowledge such as "unsigned" extends the useful range.

### 3.3 Conservative analysis: soundness, completeness, may/must

- An optimization is **sound** if it preserves the program's functionality.
- An analysis is **sound** if it never enables an optimization outside the set of scenarios where it is valid. Its
  facts over-approximate all behaviours, so it can produce **false positives** (spurious warnings) but **no false
  negatives**.
- An analysis is **complete** if it never disables an optimization that would have been valid. It
  under-approximates, so it can produce **false negatives** but no false positives. (My notes page: *over-approx ⇒
  sound ⇒ false positives; under-approx ⇒ complete ⇒ false negatives*.) A static buffer-overrun checker is
  typically **both unsound and incomplete**.
- **May** analysis (union meet) answers "on some path": RD, LV. **Must** analysis (intersection meet) answers "on
  all paths": available and very busy expressions.
  - Constant propagation is a must-style analysis: a variable is a constant only if every path agrees.
- **Safety vs liveness.**
  - Safety is the absence of bugs ("nothing bad happens"). Over-approximating static analysis is the right tool for
    proving safety.
  - Liveness, in the course's framing, is the presence of a bug or event ("something happens"). Demonstrating it
    needs a witness execution, which is where dynamic analysis is strong. Over-approximate static reports of
    "presence" can be false positives.
  - The term is unrelated to *live* variables.
- **Optimization-specific soundness** (slides 37–39). Soundness depends on the client.
  - If A computes p → {x, y, z} instead of {x, y}: O1 (replace `*p` by `x` when p points only to x) is suppressed,
    while O2 (make p volatile when it points to several variables) is enabled.
  - Approximating 1.98 as 2 enables O1 (`x*2 → x<<1`) and disables O2 (a special circuit for sums of reciprocals of
    powers of two). Approximating it as 1.96875 does the reverse.
  - **What is conservative for one client may be unsound for another.**

---

## 4. Pointer analysis (PTR)

### 4.1 Why pointer analysis is needed

Each application reduces to an alias question:

- **Dead-code elimination.** `q = &a[ii]; p = &b[jj]; if (p == q) {...} else {...}`. If the analysis proves that p
  and q never alias, the condition is always false and the then-branch is dead. Proving it requires comparing
  `a + ii*sz` with `b + jj*sz`, or `s1.arr + …` with `s2.ptr + …`.
- **Common-subexpression elimination.** `p + ii == q + jj` with `q = s1.arr; p = s1.ptr`. The two expressions are
  common only if they denote the same location.
- **Parallelization.** `f(){*p = 10;}` and `g(){*q = 20;}` can run concurrently iff `!alias(*p, *q)`.
- **Escape analysis.** A definition through p escapes f if `pointsto(p, x)` for some x that is a global or a heap
  object. If p points only to locals, they can be stack-allocated or optimized aggressively.

### 4.2 The four constraint forms (flow-insensitive, C)

Heap allocation `p = malloc()` is treated as `p = &heap_site`.

| Statement | Name | Constraint | Read set | Write set |
|---|---|---|---|---|
| `p = &q` | address-of | q ∈ pts(p) | {} | {p} |
| `p = q` | copy | pts(q) ⊆ pts(p) | {q} | {p} |
| `p = *q` | load (complex) | ∀v ∈ pts(q): pts(v) ⊆ pts(p) | {q} ∪ pts(q) | {p} |
| `*p = q` | store (complex) | ∀v ∈ pts(p): pts(q) ⊆ pts(v) | {q, p} | pts(p) |

(The read/write sets come from the CC'12 deck and are used for conflict detection in parallel PTA, §4.8.)

**How each form appears in -O0 LLVM IR** (the A3 lesson):

| C | IR pattern |
|---|---|
| `p = &x` | `store ptr %x, ptr %p` (x is an alloca **or a GlobalVariable `@x`**) |
| `p = q` | `%1 = load ptr, ptr %q` ; `store ptr %1, ptr %p` |
| `p = *q` | `%1 = load ptr, ptr %q` ; `%2 = load ptr, ptr %1` ; `store ptr %2, ptr %p` |
| `*p = q` | `%1 = load ptr, ptr %q` ; `%2 = load ptr, ptr %p` ; `store ptr %1, ptr %2` |
| `*p = &x` | `%1 = load ptr, ptr %p` ; `store ptr %x, ptr %1` (a store whose source is an address) |
| `*p = *q` | needs a temporary: `t = *q; *p = t` |

### 4.3 Andersen's analysis (inclusion-based, 1994 PhD thesis)

- Solve the subset constraints to a fixpoint. Worst case is **O(n³)**: n nodes, O(n²) edges, sets of size n.
- It is the most precise flow- and context-insensitive analysis. Its result is the *least* solution of the
  constraints.
- **Worked examples** (all verified with a solver; `scratchpad/pa_ptr/pta.py` in the Oct 2026 session):
  - A3 spec example `b=&x; a=b; b=*h; h=a` → a→{x}, b→{x}, h→{x}. So `pts(a) ∩ pts(b) = {x}` and a and b alias.
  - Graph-rewrite example `a=&x; b=&y; p=&a; c=b; *p=c` → a→{x,y}, b→{y}, c→{y}, p→{a}.
  - Constraint-graph classwork `a=&x; b=a; *p=b; p=&a; *q=p; b=&y; a=&b; q=b` → a, b, q → {a,b,x,y}; p → {a};
    x, y → {a}.

### 4.4 Steensgaard's analysis (unification-based, POPL'96)

- **Idea.** Every abstract location (ECR, an equivalence class) has **at most one pointee ECR**. Assignments
  **unify** instead of adding subset edges:
  - `p = &q` → join(\*p, q)
  - `p = q` → join(\*p, \*q)
  - `p = *q` → join(\*p, \*\*q)
  - `*p = q` → join(\*\*p, \*q)
  - Joining two ECRs recursively joins their pointees.
- With union-find the cost is **O(n·α(n))**, almost linear. The result is a **points-to hierarchy** (a graph of
  ECRs with out-degree ≤ 1). It is less precise than Andersen because targets of the same pointer get merged.
- **Examples (verified):**
  - `a=&a; b=*x; c=&a; b=a; c=x` (MidSem 2014 Q7) → {a}→{a}, b→{a}, c→{a}, x→{a}. Here x gets a pointee only by
    unifying with c.
  - `*b=a; q=*c; c=*a; b=&q; b=&c; a=&a` → {a}→{a}, b→{c,q}, {c,q}→{a}. Andersen gives a→{a}, b→{c,q}, c→{a},
    q→{a}.
- **Prof. George's claim (MidSem 2016 Q5)** — "add Q=P for every P=Q, run Andersen, and you get Steensgaard" — is
  **false**.
  - Counterexample (verified): `a=&x; a=&y; x=&z;` contains no copy, load or store, so nothing is mirrored and
    Andersen gives y → {}.
  - Steensgaard unifies x and y because both are pointees of a, so y → {z}.
  - Bidirectional copies equate the pointer sets but never merge the *targets*.

### 4.5 Constraint-graph formulation

- **Nodes** are pointers. An **edge q → p** means pts(q) ⊆ pts(p).
- **Algorithm:**
  ```
  process address-of constraints (initial points-to sets)
  add edges for copy constraints
  repeat
    propagate points-to sets along edges
    add edges from load/store constraints using current pts
  until fixpoint
  ```
- **Slide example** (initial a→{a,q,r,s,t}, p→{b,c,d}; constraints `*e=c, c=*a, e=d, b=a, *a=p`):
  - Iteration 1 adds b ⊇ a.
  - Iteration 2: `*a=p` puts {b,c,d} into q, r, s and t, and `c=*a` makes c → {a,b,c,d,q,r,s,t}.
  - Iterations 3–4 reach d and e.
  - At the fixpoint (iteration 5) everything except p points to {a,b,c,d,q,r,s,t}, and p → {b,c,d}. Verified.
- **Why the graph formulation?** Naively it is no better than the constraint form. The win comes from exploiting
  the graph's **structure**:
  - **Cycles.** For `a=b, b=a`, or `a=b, b=c, c=a`, every node on a cycle of subset edges has the **same**
    points-to set at the fixpoint, because the inclusions become equalities. Collapsing the SCC into a
    representative (union-find) therefore **loses no precision**. This is the classic exam question.
    - Cycles also appear dynamically, e.g. `a = c, b = *p, c = b` once p → a.
    - Hence **online cycle detection** (Lazy and Hybrid Cycle Detection, Hardekopf & Lin, PLDI'07).
  - **Pointer equivalence.** Pointers with the same points-to set can share one representative and be tracked
    once.
    - **Offline variable substitution (OVS)** finds some of these before solving. Examples: `a=b, b=a`;
      `a=*p, *p=a`; `a=b, c=a, c=b` when c has no other incoming edge.
  - **Dominators (Nasre, ISMM'12, "Exploiting the structure of the constraint graph").** If every path that
    brings information to n passes through d, then n's set equals d's, so a dominator and its dominees are pointer
    equivalent. Dominance is transitive (a doms b, b doms g ⇒ a doms g).
    - The paper maintains this with incremental dominator updates.
    - It is 88% faster than BDD-based Lazy Cycle Detection and 2× faster than Deep Propagation.
  - **Propagation order.** Topological order is *wave propagation*. Depth-first order is *deep propagation*, which
    pairs well with **difference propagation**: only push the newly added part of a set.

### 4.6 Constraint ordering is NP-complete; prioritized PTA

- Finding the optimal evaluation order is NP-complete, by reduction from **Set Cover** SC(U, S, K).
  - Each set S_i becomes a variable with pts(S_i) = its elements.
  - Add copy statements `S = S_i`.
  - "Can S reach the fixpoint (cover U) in K steps?" holds iff K sets cover U.
  - The reduction is linear and solutions verify in polynomial time, so the problem is NP-complete.
  - Example: S = {1,4}, {2,5}, {2,4,5}, {3} → the cover {1,4}, {2,5}, {3} needs 3 steps.
- Hence heuristics. **Prioritized points-to analysis (Nasre):**
  - The priority of a constraint in iteration i is the **amount of new points-to information it added in
    iteration i−1**.
  - Constraints are bucketed into priority levels and processed high to low. A constraint can jump levels.
  - *Skewed evaluation*: high-priority buckets are evaluated more often.
  - In the slide's example, processing `*a=p (18)`, `c=*a (8)`, `*e=c (0)` in priority order reaches the fixpoint
    in **3** iterations versus **4** for Andersen's fixed order.

### 4.7 Design dimensions for PTA

- Sensitivity: flow, context, field.
- Heap modelling: allocation sites, or allocation site plus calling context.
- Strong vs weak updates. A strong update (kill) needs a must-alias to a single concrete location; flow-insensitive
  analyses only do weak updates.
- Whole-program vs demand-driven (CFL-reachability).
- Representation: bitvectors, sparse sets, BDDs.

### 4.8 Parallel points-to analysis (Putta & Nasre, CC'12)

- **Conflicts.** Two constraints conflict if one writes a variable that the other reads or writes, using the
  read/write sets in §4.2.
  - Naive pipeline: find conflicts (costly), schedule, analyze in parallel, update.
  - On the 15-constraint example: **13 steps sequential, still 9 with 8 cores**. Store constraints (writes to
    `pts(p)`) are the main culprits.
- **Replication-based approach:**
  - Each thread reads from the **master copy**, writes to a **local replica**, then merges into the master.
  - Partitioning is arbitrary.
  - On the example: 3 threads take 12 steps, 4 threads 9 steps, 5 threads 8 steps.
- **Why replication works:**
  1. The computation is **monotonically increasing**: points-to sets never shrink, so merging is union.
  2. The algorithm is **unordered**: constraints can be processed in any order and the least fixpoint is unique.
  - "A naïve replication doesn't work for flow-sensitive analysis." There, per-point facts are *recomputed*
    (out = gen ∪ (in − kill)), so overwriting with a stale replica can regress a fact.
- **Naive vs replication:**

  | Naive | Replication |
  |---|---|
  | No merging | Costly merging |
  | Fewer iterations | More iterations |
  | General purpose | Needs monotone, unordered |
  | Lower memory | Higher memory |
  | Costly conflict detection | None |
  | Limited parallelism | Adaptive |
  | Unbalanced load | Better load balance |
  | Lower parallel performance | Better parallel performance |
- **Optimizations:**
  - load balancing (*orphan-and-adopt*);
  - parallel online cycle elimination (disjoint cycles collapse in parallel);
  - reducing replication cost (single writer, difference propagation, constraint affinity);
  - limited scheduling.
- **Results:** on 16 SPEC 2000 plus 5 open-source programs at 4–8 threads, it beats the sequential baseline and
  the Galois implementation (Méndez-Lojo, Mathew & Pingali, OOPSLA'10).

### 4.9 PTA as graph rewrite rules

- The approach evolved in three steps:
  1. Constraint sets: pointers with points-to sets.
  2. A graph: pointers as nodes, subset edges, a set attached to each node.
  3. **Graph rewrite rules**: *variables* as nodes and *all* relations as edges (points-to, copy, load, store). The
     points-to set is **implicit**: the targets of p's points-to edges.
- The rules (with m, n, o as nodes):
  - **copy rule:** m ←copy n and n →pts o ⇒ add m →pts o.
  - **load rule:** m ←load n and n →pts o ⇒ add m ←copy o.
  - **store rule:** m →store n and n →pts o ⇒ add o ←copy m.
- Rules can fire **in parallel** wherever their patterns match (amorphous data-parallelism). Open questions: how
  to order rule evaluation, and how to combine rules for efficiency.
- **Classwork** `*p=c; b=&y; b=*p; p=&a; a=&x; *p=c; c=p; c=&z` → **a→{a,x,z}, b→{a,x,y,z}, c→{a,z}, p→{a}**
  (verified; matches the slide).

---

## 5. Shape analysis (SHA) — Ghiya & Hendren, POPL'96

**Shape analysis** statically identifies the structural (topological) properties of heap data structures.

### 5.1 Why points-to analysis is not enough

- **List reversal:** `for (y = null; x; ) { t = y; y = x; x = x->next; y->next = t; }`. We want to prove that x is
  still an acyclic singly-linked list at the end.
- The heap is unbounded, so points-to analysis needs a **summary node**: node 1 is concrete and 2\* stands for all
  remaining nodes.
- After a few iterations, `t, x, y, 1.next, 2*.next → {null, 1, 2*}`: "everything can point to everything".
  Acyclicity is lost even though the structure stays a list.
- Fix: keep extra **structural** facts (e.g. "2\* is acyclic") and use them, for instance "removing a node keeps
  acyclicity".

### 5.2 Shapes

- **Tree:** a unique path between any two nodes.
- **DAG:** acyclic, with possibly multiple paths to a node (shared substructure).
- **Cycle:** at least one cycle.
- Precision decreases from slist to tree to DAG to cycle.
- A pointer's shape is the shape of the **structure reachable from that pointer**, not of the whole heap. Shapes
  must be derived from program variables only, because the heap cannot be represented.
- Classwork:
  - `p = malloc; q = p; q->data = 20; q->next = NULL` is a tree (one node shared by p and q).
  - `q->next = p` creates a cycle of length 1.
  - `p = createTree(); q->left = p->right` gives a DAG (shared structure).

### 5.3 Abstractions

For each pair of heap-directed stack pointers:
- **D[p][q] (direction):** p's node has a path to q's node. D[p][p] = 1 iff p points to a heap object.
- **I[p][q] (interference):** p and q can reach a common node. I is symmetric; I[p][p] = 1 iff p is non-null.
- **A[p] (shape attribute):** Tree, DAG or Cycle, ordered Tree < DAG < Cycle. The merge operator ⊔ takes the max.

Why both D and I? Shape plus reachability is not enough. With s, t, q trees and `p = q->f`, after `x = t->g;
x->g = q->f` we need to know whether t's and q's structures **share** nodes. That is interference.

Shape is abstracted *separately* from D. Deducing shape from D after `p = q->f` would lose Tree/DAG. The
implementation must build **new** D/I matrices from the old ones; updating in place is unsound or imprecise.

### 5.4 Exact inference rules (from the paper; H is the set of heap-directed pointers)

**1. `p = malloc()`**
- Kill: all D(p,s), D(s,p) and I(p,s).
- Gen: D(p,p) and I(p,p).
- A[p] = Tree.

**2. `p = q` (also `p = &(q->f)` and `p = q op k`, so the analysis is not field-sensitive here)**
- Kill: as for malloc.
- D_gen_from = { D(s,p) | s ≠ p ∧ D[s,q] }
- D_gen_to = { D(p,s) | s ≠ p ∧ D[q,s] } ∪ { D(p,p) | D[q,q] }
- I_gen = { I(p,s) | s ≠ p ∧ I[q,s] } ∪ { I(p,p) | I[q,q] }
- A[p] = A[q].
- The slide notes that Dasgupta, Karkare and Reddy (IITK) address the field-sensitivity loss.

**3. `p = NULL`**
- Kill: as for malloc. No gen.
- A[p] = Tree (default).

**4. `p = q->f`**
- Kill: as for malloc.
- D_gen_from = { D(s,p) | s ≠ p ∧ **I[s,q]** }. Anything interfering with q might reach p's node; this is the
  reason I exists.
- D_gen_to = { D(p,s) | s ≠ q ∧ s ≠ p ∧ D[q,s] } ∪ { D(p,q) | A[q] = Cycle } ∪ { D(p,p) | D[q,q] }.
  - The analysis does not know *which* field leads where, so p conservatively reaches everything q reaches, except
    q itself unless the structure is cyclic.
- I_gen = { I(p,s) | s ≠ p ∧ I[q,s] } ∪ { I(p,p) | I[q,q] }
- A[p] = A[q]. A sub-piece of a tree is a tree; if A[q] = Cycle we must say Cycle.

**5. `p->f = NULL`**
- No kill and no gen: we do not know which relationships went through f.
- Attributes are left unchanged, so a structure that was temporarily DAG or Cycle stays that way: *once cyclic,
  always cyclic*.

**6. `p->f = q` (the "nasty" statement)**
- No kill.
- D_gen = { D(r,s) | D[r,p] ∧ D[q,s] }
- I_gen = { I(r,s) | D[r,p] ∧ I[q,s] }
- Shape, evaluated on the old matrices:
  - If **D[q,p]**: a cycle forms, and every s with D[s,q] ∨ D[s,p] gets A[s] = Cycle.
  - Else if **A[q] = Tree**: every s with I[s,q] ∧ D[s,p] gets A[s] = A[s] ⊔ DAG (shared structure). If p's and
    q's structures were disjoint, nothing changes.
  - Else (A[q] = DAG or Cycle): every s with D[s,p] gets A[s] = A[s] ⊔ A[q].

### 5.5 Worked exam questions

Setup for both: p1 and p2 point to the same node, p3 to another node, and all are Tree. Initially D[i][i] = 1,
D[p1][p2] = D[p2][p1] = 1, I is all ones within {p1, p2}, and I[p3][p3] = 1.

**`p3 = p2->next`** (MidSem 2014 Q8; rule 4 with p = p3, q = p2)
- Kill everything involving p3.
- from: s with I[s,p2] are p1 and p2, so D[p1][p3] = D[p2][p3] = 1.
- to: s with D[p2,s], s ∉ {p2, p3}, is p1, so **D[p3][p1] = 1**. This is spurious, but it is what the rule
  generates. D[p3][p3] = 1. No D[p3][p2], because A[p2] = Tree.
- I: I[p3][p1] = I[p3][p2] = I[p3][p3] = 1.
- Shapes: all Tree (A[p3] = A[p2]).
- Result:

  | D | p1 | p2 | p3 |
  |---|---|---|---|
  | p1 | 1 | 1 | 1 |
  | p2 | 1 | 1 | 1 |
  | p3 | 1 | 0 | 1 |

  I is all ones.

**`p1->next = p3`** (MidSem 2017 Q8; rule 6 with p = p1, q = p3)
- D_gen: r with D[r,p1] are p1 and p2; s with D[p3,s] is p3. So D[p1][p3] = D[p2][p3] = 1.
- I_gen: I[p1][p3] = I[p2][p3] = 1 (symmetric).
- Shape: D[p3][p1] = 0, so no cycle. A[p3] = Tree, and the s with I[s,p3] ∧ D[s,p1] in the *old* matrices form
  the empty set, so all stay Tree.
- Rows of D: p1 = (1,1,1), p2 = (1,1,1), p3 = (0,0,1). I is all ones.

**Queries (EndSem 2016 Q2):**
- (a) Same node: in an acyclic shape, D[p][q] ∧ D[q][p] ⇒ same node. In a Cycle it cannot be concluded.
- (b) Ancestor in a rooted DAG: D[p][q] = 1 ∧ D[q][p] = 0. D is "may", so 0 is definite and 1 means possibly.
- (c) Common ancestor: D[r][p] ∧ D[r][q].

**"Reachable shape, not actual shape" (EndSem 2016 Q3).** Take p → n1 → n2 and q → n3 → n2. Seen from p it is a
tree, and seen from q it is a tree, but the overall structure is a DAG (n2 has two parents). Ghiya–Hendren reports
A[p] = A[q] = Tree, which is correct for its definition.

**Integer D/I (EndSem 2016Jan Q3).**
- Store **path counts** instead of booleans: D[p][q] = number of distinct paths.
- Benefits:
  - D = 1 identifies trees exactly, and D ≥ 2 signals sharing. That distinguishes Tree from DAG locally.
  - Killing becomes possible for `p->f = NULL` if we know the count contributed through f.
- Trade-offs: counts can grow unboundedly in loops (need saturation at k, i.e. k-limiting), cost more memory and
  time, and still cannot attribute paths to fields without field-sensitivity.

**Improving Andersen with shape (MidSem 2017 Q9).**
- If shape analysis says the structure from x is a **Tree** and p, q are at different positions in it (D[p][q] = 0
  and D[q][p] = 0, and the nodes are distinct), then `*p` and `*q` cannot alias. Andersen's summary node would say
  they may.
- Concretely: in a loop over a list, `p = p->next` keeps p and its successor distinct, so updates through p cannot
  affect what `p->next` points to. Use this to kill or partition summary-node points-to edges.

### 5.6 Summary and improvements

The analysis is simple and practical (McCAT C compiler) but loses precision on structure updates. Improvements:
field-sensitivity, better heap modelling (TVLA, the 3-valued logic of Sagiv, Reps & Wilhelm, TOPLAS'02), and path
sensitivity. "We are still far from a precise and scalable analysis."

---

## 6. Parallelization (PAR)

### 6.1 Speedup and Amdahl

- Speedup = T_seq / T_par.
- **Amdahl:** S = 1 / ((1 − p) + p/S_p), where p is the parallel fraction and S_p the speedup of the parallel
  part. As S_p → ∞, S ≤ 1/(1 − p).
  - **20% sequential caps speedup at 5×**, irrespective of cores or effort.
  - It assumes a fixed problem size (Gustafson's law relaxes that; see ID5130 notes).
- Annotation: T_total(N) = T_setup + T_compute(N) + T_finalization. Only T_compute shrinks with N.
- **Instruction-level vs data-parallel vs task-parallel.**
  - In `S1: a[i]=2; S2: b[i]=4; S3: c[i]=a[i]`, S1 and S2 are independent. S1 → S3 is a dependence (S1 is the
    *source*, S3 the *sink*).
  - Data-parallel means the same operation on different data.
  - Task parallelism generalises to arbitrary control and data.

### 6.2 Dependences

- **Control dependence:** `if (x == 4) y = 10; else y = 1;`. The assignments depend on the condition.
- **Data dependence:**
  - **true/flow (RAW)** S1 δ S2: `x = …; … = x`
  - **anti (WAR)** S1 δ⁻¹ S2: `… = x; x = …`
  - **output (WAW)** S1 δ° S2
  - **input (RAR)**, harmless.
  - WAR and WAW are *false* (name) dependences, removable by **renaming** or privatization. RAW is real. RAW, WAR
    and WAW all lead to data races.
- **Program order vs dependence order.**
  - Program order is a total order and too restrictive. Only the partial order of dependences must be kept:
    *reorder flow, maintain dependence*.
  - Reordering buys:
    - locality: spatial (matrix ops), and temporal (`xinit(); xcompute(); yinit(); ycompute()`);
    - load balance (`small1(); big1(); small2(); big2()` → pair big with small);
    - parallelism (`xuse(); xdef(); yuse(); ydef()`).
- **Loops.**
  - The iteration vector is (i_outer, …, i_inner). The iteration space is the set of all vectors. S(iv) is a
    statement instance.
  - **S(iv) δ S(jv)** iff:
    - (a) iv < jv lexicographically, or iv == jv with a path from S to S′ in the body;
    - (b) both access the same location;
    - (c) at least one access writes.
- **Reordering transformations** change only execution order; they add or delete no executions. A reordering
  **preserves a dependence** if it keeps the relative order of source and sink.
  - **Theorem:** a reordering that preserves every dependence yields an equivalent computation.
  - A transformation is **valid** if it preserves all dependences, and **safe** if it yields an equivalent program.
    Valid ⇒ safe, but not conversely.
  - Classwork: an equivalent program that breaks a dependence. `t = x*2; y = t; …` reordered with a recomputation
    such as `y = x*2` gives the same output but no longer has the RAW on t.
- **Loop parallelization theorem:** converting a sequential loop to parallel is valid if the loop **carries no
  dependence**.
  - `for k: S1: a[k]=b[k]; S2: b[k]=a[k]+1;` has dependences only *within* an iteration, so it is parallel.
  - `a[k] = a[k+1]` carries one.
  - `a[i][j] = a[i][j-1]*pi` carries along j only, so the **outer loop** (rows) is parallel.
  - `a[i][j] = a[i-1][j-1]*pi` is parallel along **anti-diagonals** (wavefront).

### 6.3 Dependence testing as integer linear programming

- For `a[f(i,j)][g(i,j)] = … a[h(i,j)][k(i,j)] …` with **affine** f, g, h, k (a linear function of loop variables
  plus a constant; `a[i*j]` and `a[b[i]]` are not affine), a flow dependence from (iw, jw) to (ir, jr) exists iff
  the following system has an **integer** solution:
  - loop bounds for both iterations;
  - ordering (iw, jw) < (ir, jr), which is a disjunction, one ILP per case: iw < ir, or iw = ir ∧ jw < jr;
  - same location: f(w) = h(r) and g(w) = k(r), each written as two ≤ inequalities.
- **Example (1-D).** `for ii<10: a[2ii] = … a[2ii+1] …`.
  - Flow: 0 ≤ iw ≤ ir−1, ir ≤ 9, 2iw = 2ir+1. No integer solution (parity), so **no flow dependence**.
  - With a read of `a[ii+1]`: 2iw = ir+1 has solutions, so a **true dependence exists**.
  - Every write/read pair must be checked: 2ii vs ii+1, 3+ii vs 5ii, and so on. Write–write pairs give WAW;
    read–read pairs don't matter.
- **Nested example** `for ii<4, jj<10: a[2ii+3][jj+4] = … a[ii−1][2jj−5]`.
  - Memory: 2iw+3 = ir−1 and jw+4 = 2jr−5.
  - Same ii: 2ii+3 = ii−1 gives ii = −4, impossible, so the inner loop is parallel.
  - Different ii: ir = 2iw+4 ≥ 4 is out of range, so the outer loop is parallel too.
  - Three ways to parallelize: inner-only, outer-only, nested. In practice iterations are chunked.
- **Matrix form Ax ≤ b.** EndSem 2014 Q2: `for (i=4; i<=N; ++i) a[i−3] = a[2i+3] − 4;`, true dependence.
  - Variables x = (iw, ir) with N symbolic:
    ```
    [-1  0]          [-4]      (iw ≥ 4)
    [ 1  0]          [ N]      (iw ≤ N)
    [ 0 -1]  [iw]    [-4]      (ir ≥ 4)
    [ 0  1]  [ir] ≤  [ N]      (ir ≤ N)
    [ 1 -1]          [-1]      (iw < ir: write before read)
    [ 1 -2]          [ 6]      (iw − 3 = 2ir + 3, as two inequalities)
    [-1  2]          [-6]
    ```
  - Infeasible: iw = 2ir + 6 > ir. So there is **no RAW**, but there is a **WAR** (the read at ir precedes the write
    at 2ir + 6). That loop-carried anti-dependence needs privatization or a copy before parallelizing.
- **WAW, MidSem 2016 regular Q6:** `for i∈[0,9], j∈[1,4]: a[2i+3j−4] = 5*a[6i+7j−8] − 9`.
  - Unknowns (i1, j1, i2, j2) with bounds, 2i1+3j1 = 2i2+3j2, and ordering (i1 < i2) or (i1 = i2 ∧ j1 < j2) as two
    systems.
  - Feasible, e.g. (0,3) and (3,1): 9 = 9. So **WAW exists**.
- **RAW, MidSem 2016 self-study Q6:** `for i∈[1,9], j∈[i+1, i+9]: a[i+3j−1] = a[2i−8]`.
  - Bounds include the coupled i+1 ≤ j ≤ i+9 (a non-rectangular space is fine for ILP).
  - Constraints: iw+3jw−1 = 2ir−8 and (iw,jw) < (ir,jr).
  - Feasible: (1,2) writes a[6], and ir = 7 reads a[6]. So **RAW exists**.

### 6.4 GCD test

- A linear Diophantine equation Σ aᵢxᵢ = c has an integer solution **iff gcd(aᵢ) divides c**. It ignores bounds, so
  it can only prove *independence*: "dependence may exist" otherwise.
- `arr[4ii] = …; … = arr[8ii+3]` gives 4i1 − 8i2 = 3. gcd = 4 does not divide 3, so **no dependence**. With
  `arr[8ii+4]`, 4 | 4, so it **may** exist.
- **Slide classwork, corrected.** `for (ii=1; ii<6; ++ii) { b[ii] = a[3ii−5] + 2.0; a[2ii+1] = 1.0/ii; }`.
  - 3i2 − 2i1 = 6, and gcd = 1 divides 6, so a dependence may exist.
  - The slide then says "due to ii < 6, true dependence does not exist", but **(i1, i2) = (3, 4)** is in range:
    the write a[7] at ii=3 is read at ii=4.
  - So a **true dependence does exist**. Check this whenever you reuse the slide.
- Stronger tests: Banerjee (bounds), Omega test (Pugh, exact integer Fourier–Motzkin), polyhedral libraries (isl).

### 6.5 Managing races

- Choices: locks (critical sections), atomics, barriers.
- **Locks.**
  - `if (i == p || i == q) { lock(f(i)); op; unlock(f(i)); }`.
  - Producer–consumer: a single producer adds items; multiple consumers remove them.
  - Multiple locks give more parallelism but risk **deadlock** (each waits for the other's lock). Deadlock
    avoidance can cause **livelock** (threads keep retrying without progress). Either may be tolerated if rare.
- **Atomics** suit simple ops on a primitive single element: relative updates and read-modify-write.
  - atomicInc/Dec/Add/Sub/Min/Max/Exch/CAS.
  - Lower overhead than locks, and no deadlock.
  - `sum += a[i]` → `atomicAdd(&sum, a[i])`.
  - Classwork: a lock-free singly-linked-list insert is a CAS on `prev->next`.
- **Barriers.**
  - All threads wait at a synchronization point. The effect is like **loop distribution**.
  - May add more dependences than required but must keep the existing ones. Removing a barrier (between
    `items.remove` and `items.add` phases) changes behaviour: new items from one thread become visible to others
    within the round.
  - Rule of thumb: sparse dependences → atomics or locks; dense → barriers.
- **Worklist via barriers + atomics:**
  ```
  item = items.remove();  atomicDec(items.size)
  --- barrier ---
  moreitems = process(item)
  --- barrier ---
  atomicAdd(items.size, size(moreitems)); items.addParallel(moreitems)
  ```
- **"PAmistery":**
  ```
  do { if (gg < K) { old = atomicInc(&gg); if (old >= K) atomicDec(&gg); } } while (old >= K);
  // critical section
  cleanup() { atomicDec(&gg); }
  ```
  This is a **counting semaphore**: at most K = 29 threads in the critical section. Use it for resource or
  thread-pool limiting, e.g. bounding concurrent connections or GPU streams.

### 6.6 Limits of static parallelization: irregular programs

- **Irregular** means data access or control flow depends on runtime data: `c[5] = b[a[4]]`,
  `if (a[4] > 30)`, pointer-based structures, graphs. Examples: points-to analysis itself, Delaunay mesh
  refinement, MST, survey propagation, shortest paths.
- Matrix multiply is regular: its behaviour is known from sizes alone.
- **Static analyses are too conservative here**, which forces pessimistic synchronization. That motivates dynamic
  techniques (Pingali's **Tao of parallelism / Galois**):
  - an **active node** and its **neighbourhood**;
  - non-overlapping neighbourhoods run in parallel; overlapping ones need synchronization;
  - **optimistic** (speculate and roll back) or **cautious** (acquire the whole neighbourhood before writing)
    execution.
- Sequential programs over-specify order (`for ii: process(a[ii])`, `while (m<n) { process(m); m = next(m); }`).
  `forall (e in a) process(e)` declares **unordered** execution. It adds unorderedness, **non-determinism** and
  higher-level information.

---

## 7. Program slicing (SLI)

### 7.1 Definitions

- A **slice** is the subset of statements (possibly) relevant to an event of interest (e.g. a segfault at `=*p`).
- Applications: program understanding and debugging, restructuring (splitting `lineCharCount` into line-count and
  char-count programs), program differencing, test coverage, model checking.
- **Kinds:**
  - **Backward** slice at p: what may *affect* p.
  - **Forward** slice: what may *be affected by* p.
  - **Chop**(p, q): affected by p **and** affecting q.
- **Static** slicing holds for all inputs, assumes the program halts, and need not be unique. The whole program is
  always a slice. A statement-minimal slice is uncomputable (halting problem).
- **Dynamic** slicing works on an execution trace for a given input. It is more precise, and its criterion includes
  the statement *instance* and the input.

### 7.2 Weiser's data-flow equations (ICSE'81 / TSE'84)

For criterion C = ⟨i, V⟩:
- **Directly relevant variables.** R⁰_C(n) contains v if:
  1. n = i and v ∈ V; or
  2. n has a successor m and either
     - (2a) v ∈ USE(n) and DEF(n) ∩ R⁰_C(m) ≠ ∅, so a relevant variable is defined here and its operands become
       relevant; or
     - (2b) v ∉ DEF(n) and v ∈ R⁰_C(m), so relevance passes through.
  - Example `1: y = x; 2: a = b; 3: z = y;` with ⟨3, y⟩: R(3) = {y}, R(2) = {y} by 2b, R(1) = {x} by 2a.
- **Directly relevant statements:** S⁰_C = { n | DEF(n) ∩ R⁰_C(n+1) ≠ ∅ }.
- **Control dependence:**
  - INFL(b) is the set of statements whose execution b directly controls.
  - B⁰_C = { b | INFL(b) ∩ S⁰_C ≠ ∅ }.
  - Each such branch becomes a new criterion BC(b) = ⟨b, USE(b)⟩.
- **Iterate:**
  - R^{i+1}_C(n) = R^i_C(n) ∪ ⋃_{b∈B^i_C} R⁰_{BC(b)}(n)
  - S^{i+1}_C = { n | DEF(n) ∩ R^{i+1}_C(n+1) ≠ ∅ } ∪ B^i_C
  - stop at the fixpoint.
- **Slicing as DFA need not be minimal.**
  - Program: `1: a=1; 2: while(f(k)) { 3: if (g(c)) { 4: b=a; 5: x=2; } else { 7: c=b; 8: y=3; } 9: k=k+1; } 10: z=x+y;`
  - Criterion ⟨10, z⟩.
  - Is statement 1 in the slice? The data-flow version includes 1 and 4 through `c = b`, which affects the
    condition g(c) on the *next* iteration.
  - The deck's point: a smarter analysis that reasons about which path executes can drop them; data-flow slicing
    over-approximates.

### 7.3 Dynamic slicing

On input n=2 for `1: read(n); 2: i=1; 3: while (i<=n) { 4: if (i%2==0) 5: x=7; 6: else 7: x=16; 8: ++i; } 9: write(x);`:
the assignment `x=16` from the first iteration is killed by `x=7` in the second. So the dynamic slice for ⟨9, x⟩
**omits the else branch**, while the static slice keeps it.

| Static | Dynamic |
|---|---|
| All inputs | Specific inputs |
| Less precise | More precise |
| Works on source | Works on source or trace |
| Criterion: statement + variables | Criterion: statement instance + variables + input |

### 7.4 Dependence graphs

- **CDG (control dependence graph).** Y is control dependent on X if X decides whether Y executes. Formally, Y
  post-dominates a successor of X but not X itself. A CDG is not a CFG.
- **DDG:** def–use edges.
- **PDG** = CDG + DDG (Ferrante, Ottenstein & Warren, TOPLAS'87). A **slice is backward (or forward) reachability
  in the PDG**. Interprocedural slicing uses the **SDG** (Horwitz, Reps & Binkley 1990) with calling-context-aware
  reachability.
- **Syntactically valid slices** may need extra statements, e.g. keeping an `if` header so that `x = x+1` still
  parses. That is easier in C because `;` terminates statements. A syntactically valid slice can still be
  semantically wrong.

### 7.5 Worked slices from past papers

- **MidSem 2016 Q5**, criterion ⟨13, z⟩ for the scanf/malloc/two-loop program: z needs lines 12, 9, the loop at 10,
  y (11, 8), a's contents (6, inside loop 5 with 7), a (4) and n (3). The slice is **{3, 4, 5, 6, 7, 8, 9, 10, 11,
  12, 13}**: everything except the declarations, which are kept for validity.
- **EndSem 2014 Q6**, criterion ⟨11, sum⟩ for `2: sum=0; 3: i=0; 4: j=1; 5: while(i<N){ 6: i=++j; 7: sum+=i; 8: i=sum/2; } 10: if (sum>=100) 11: printf(sum); 12: else 13: printf(...)`.
  The slice is **{2, 3, 4, 5, 6, 7, 8, 10, 11}**.
- **EndSem 2016 Q4**, forward slice ⟨2, x⟩ (`int x = 0`) in the x/n/a/i program: **{2, 10, 12, 16}**. x is only
  updated by `x += a`, `x -= a` and printed.
- **EndSem 2017 Q7**, forward slice ⟨2, i⟩ with `j = &i`:
  - 4 (`*j < 4` reads i) is included, and so are 5, 6, 8 and 9, which are control dependent on 4.
  - 11 (flag) is included, then 12 and 14, which are control dependent on 11.
  - 15 is included.
  - The slice is **{2, 4, 5, 6, 8, 9, 11, 12, 14, 15}**. Line 3 takes i's *address*, not its value, so it is
    excluded.

---

## 8. Security analysis (SEC)

### 8.1 Vulnerabilities

- **Effects:** crash, non-termination, wrong output, unintended actions.
- **Causes:** dangling pointers, buffer overruns, null dereference, wrong opcode, arbitrary data change,
  self-modifying code, hardware issues.
- **C is susceptible** because it allows direct pointer manipulation (space and performance first, not security)
  and its standard library has unsafe functions (`gets`, `strcpy`, `strcat`).
  - `strncpy` does **not** guarantee null termination, and it pads with zeros.
- **Stack smashing.** `f(char *b){ gets(b); }` called with `char a[4]` in main. Writing past `a[3]` overwrites the
  saved return address (L1), so `pop PC` jumps to attacker-chosen code.
- **Defences:**
  - **Canary/sentinel** next to the return address, checked before returning (randomized; no guarantee).
  - Explicit bounds checks (Java).
  - Wrapping system calls with checks.
  - Dynamic techniques cost runtime overhead and terminate the program.
  - **W^X**: writable code invites self-modifying exploits.
  - Heap smashing has no return addresses but does have **function pointers**.
- **gcc demo:** `char buff[15]; int pass = 0; gets(buff);`. Typing 20 h's sets `pass` and grants root on old gcc.
  New gcc prints `*** stack smashing detected ***`; `-fno-stack-protector` brings the bug back.
- **Quine:** `char*f="char*f=%c%s%c;main(){printf(f,34,f,34,10);}%c";main(){printf(f,34,f,34,10);}` prints itself.

### 8.2 Static buffer-overrun detection

- It is **unsound and incomplete**: it misses risks (false negatives) and flags non-issues (false positives).
- **Annotations as pre/post-conditions** (Splint/LCLint style):
  - `minDef/maxDef/minUse/maxUse`, `notNull`, `null`, `restrict`.
  - `strcpy`: `@requires maxDef(s1) >= maxDef(s2)`, `@ensures maxUse(s1) == maxUse(s2) and result == s1`.
  - `malloc`: `@ensures maxDef(result) == size or result == null`.
- **`restrict` homework.** In `void add(int *restrict a, int *restrict b, int *restrict c, int n){ for i: c[i] = a[i] + b[i]; }`
  the compiler may keep values in registers, reorder, and **vectorize** (SIMD), because no store through c can
  change a or b.
- **Inferring constraints:**
  - from `for` init, bound and increment;
  - from array declarations and `malloc`;
  - from conditional checks.
  - A few heuristics cover most code. Inference is feasible when loop bounds don't depend on array values:
    `while (i < n)` is OK, `while (a[i] != '\0')` is not.
  - Generating *all* constraints is expensive, so past work used flow-insensitive constraints.
- **Precision needs interprocedural analysis.** `a = malloc(N); ii = N/2 + f(N); a[ii] = 0;` with `f(N) = N % 5`
  needs f's range to validate the access. Domain knowledge about N filters false positives.
- **Other targets:**
  - null dereference;
  - dangling pointers;
  - buffer overflow;
  - **stream safety** (fread without fopen).
- **Each target as a DFA:** choose the facts, the statements of interest, the direction and dimensions, the meet,
  and the transfer functions.

### 8.3 Polyhedral bounds checking (Ax ≤ B)

- **Variables:** the loop index and the symbolic unknowns (sizes, bounds).
- Write one system per **violation**: lower (index ≤ −1) or upper (index ≥ size) for each access.
- **Feasible means an out-of-bounds access is possible.** The (integer) solution is a **witness**: concrete inputs
  that trigger it. Infeasible for all systems means proven safe.
- Use ILP or the Omega test, because rational LP solutions can be spurious. Fourier–Motzkin elimination projects
  out i to get *conditions* on the inputs.
- **EndSem 2016 Q6:** `char *a = malloc(M); for (i = x; i < 20; ++i) a[2i−3] += a[i+1];` with variables (i, x, M):
  - common rows: x − i ≤ 0, i ≤ 19
  - write underflow: 2i ≤ 2. Feasible (i = 1 when x ≤ 1, giving a[−1]).
  - write overflow: M − 2i ≤ −3, i.e. 2i − 3 ≥ M.
  - read underflow: i ≤ −2.
  - read overflow: M − i ≤ 1, i.e. i + 1 ≥ M.
- **EndSem 2017 Q3:** `for (i=x; i<y; ++i) a[2i+3] = i+9;` with the common rows x − i ≤ 0 and i − y ≤ −1:
  - underflow: 2i ≤ −4;
  - overflow: M − 2i ≤ 3.
- **EndSem 2016Jan Q6:** `int *a = malloc(size)` (in **bytes**, so ⌊size/4⌋ ints), `for (i=0; i<N; ++i) a[3i−6] = a[5];`.
  - The write underflows when 3i − 6 ≤ −1, i.e. i ≤ 1. That is **feasible whenever N ≥ 1**: a definite bug.
  - The write overflows when 4(3i−6) + 4 > size.
  - The read of a[5] overflows when size < 24.

### 8.4 Tools

| Tool | Approach |
|---|---|
| **BOON** (Wagner et al., NDSS'00) | Integer-range constraints for C string buffers; flow-insensitive, intraprocedural, pointer-insensitive |
| **CQual** (Foster et al.) | Type qualifiers (tainted/untainted) propagated by type inference; catches **format-string** bugs by type checking |
| **xg++ / metacompilation** (Engler et al., OSDI'00) | Template-driven compiler extensions; found Linux kernel bugs: untrusted data, leaks, deadlocks |
| **Eau Claire** | Theorem-prover based specification checker (ESC-style): buffer overruns, file-access races, format strings |

Self-modifying code, DOS era: a batch file rewrites itself while `CALL`ing in a loop. Single-window DOS lacked
protection and inter-process communication.

### 8.5 Hoare logic, weakest preconditions, strongest postconditions (ExtraMaterial)

- **Triple** {P} S {Q}: if P holds and S terminates, Q holds (partial correctness). Rules:
  - **Assignment:** {Q[E/x]} x := E {Q}
  - **Sequence:** {P}S1{R}, {R}S2{Q} ⊢ {P}S1;S2{Q}
  - **If:** {P∧B}S1{Q}, {P∧¬B}S2{Q} ⊢ {P} if B S1 else S2 {Q}
  - **While:** {I∧B} S {I} ⊢ {I} while B S {I∧¬B}. I is the **loop invariant**; termination needs a decreasing
    variant.
  - **Consequence:** strengthen the pre-condition, weaken the post-condition.
  - `sum` example invariant: `sum = Σ_{j<i} a[j] ∧ 0 ≤ i ≤ length`.
- **wp(S, Q)** is the weakest P with {P} S {Q} (Dijkstra 1976). "Weaker" means implied by: A ⇒ B makes B weaker,
  and `true` is the weakest.
  - wp(x := E, Q) = Q[E/x]
  - wp(S1;S2, Q) = wp(S1, wp(S2, Q))
  - wp(if B S1 else S2, Q) = (B ⇒ wp(S1,Q)) ∧ (¬B ⇒ wp(S2,Q))
- **sp(S, P)** is the strongest post-condition.
  - sp(x := E, P) = ∃x₀. P[x₀/x] ∧ x = E[x₀/x]
  - sp(if) = sp(S1, P∧B) ∨ sp(S2, P∧¬B)
- **Exam answers:**
  - wp(`x = y*2`, x < 10) = **y*2 < 10, i.e. y ≤ 4** for integers.
  - wp(`x = y/z`, x > 2) = **z ≠ 0 ∧ y/z > 2**. With C truncating integer division this is
    **(z > 0 ∧ y ≥ 3z) ∨ (z < 0 ∧ y ≤ 3z)**. Over the reals it is (z>0 ∧ y>2z) ∨ (z<0 ∧ y<2z).
  - sp(`if (y==x) y = ++x; else x = y;`, y>2 ∧ x>5):
    - then-branch gives x = y ∧ x ≥ 7;
    - else-branch gives x = y ∧ y > 2;
    - **union: x = y ∧ y > 2**.
  - sp(`if (x>0) y = x+2; else y = z;`, x>3): the else branch is infeasible, so **x > 3 ∧ y = x + 2**.

---

## 9. Dynamic analysis (DYN)

### 9.1 Basics

- **Instrumentation** is the mechanism: inserting code. It can be done on source, bytecode/IR or binaries (Pin,
  DynamoRIO), at compile, load or run time.
- **Profiling** is the goal: hot spots, time per function, allocation patterns, I/O. It can use instrumentation or
  **sampling**.
- **Applications:** bug finding (testing), data-race detection, security vulnerabilities, improving the precision
  of static analysis, input-dependent analysis.

| Static | Dynamic |
|---|---|
| Sound (over-approximate) | Incomplete (only seen inputs) |
| Imprecise | Precise |
| Input-oblivious | Input-dependent |

The trend is to **combine** them.

- **Intrusive** profiling inserts instructions (file logs, pointer targets, function timers). **Non-intrusive**
  profiling uses hardware counters or wall-clock time.

### 9.2 Path vs edge profiling

- A **path profile** counts entry-to-exit acyclic paths. On the A–F CFG the paths are ACDF, ACDEF, ABCDF, ABCDEF,
  ABDF and ABDEF.
  - Naive per-path instrumentation blows up exponentially.
- An **edge profile** counts per-edge frequencies. A path's frequency is then estimated by the **minimum edge
  frequency along it**.
  - Different path profiles can produce the **same** edge profile, so edges cannot recover paths in general.
- **Efficient edge profiling** (Ball & Larus TOPLAS'94; Knuth):
  - Instrument only the **chords**: edges not in a (maximum-weight) **spanning tree** of the CFG plus an exit→entry
    edge.
  - Derive the others by **flow conservation** (in-flow = out-flow at each node).
  - Slide: with chords ac, bc, bd, de instrumented:
    - C→D = ac + bc
    - D→F = ac + bc + bd − de
    - E→F = de
    - A→B = bc + bd
    - F→A = ac + bc + bd
  - Each instrumented edge needs its own counter.

### 9.3 Ball–Larus efficient path profiling (MICRO'96)

1. **Number paths** so that path sums are unique and dense in [0, P−1]:
   ```
   NumPaths(exit) = 1
   for v in reverse topological order:
       NumPaths(v) = 0
       for each edge v→w:  Val(v→w) = NumPaths(v);  NumPaths(v) += NumPaths(w)
   ```
2. **Choose a spanning tree.** Instrument the chords with `r += Inc(c)`, where Inc(c) is the sum of Val around c's
   fundamental cycle, with signs. This minimises the number of increments on hot edges.
3. **Runtime:** `r = 0` at entry; `count[r]++` at exit. Backedges do `count[r]++; r = 0` (paths are cut at loop
   headers).
4. **Regenerate** a path from its id by walking from entry and taking, at each node, the edge with the largest
   Val ≤ the remaining sum.

**Worked numbering for the A–F CFG** (edges A→B, A→C, B→C, B→D, C→D, D→E, D→F, E→F; out-edges taken in that
order):
- NumPaths: F = 1, E = 1, D = 2, C = 2, B = 4, A = 6.
- Val: A→C = 4, B→D = 2, D→F = 1, all others 0.
- Paths: ABCDEF = 0, ABCDF = 1, ABDEF = 2, ABDF = 3, ACDEF = 4, ACDF = 5.
- With the spanning tree {A→B, B→C, C→D, D→E, E→F}, all tree edges have Val 0. So the chords simply carry
  r += 4 (A→C), r += 2 (B→D), r += 1 (D→F), and F→A carries 0.
- The slide picks a different tree (chords AC = 0, BC = 2, BD = 4, DE = 1). Numberings are not unique; any
  numbering whose sums are distinct and dense in 0..P−1 is valid.
- Hardware-friendly: one register, add-immediates, and indexed increment.

### 9.4 Other DYN topics in the outline

- **Invariant detection** (Daikon, Ernst et al.): equality and affine relations inferred from traces.
- **Dynamic type inference.** These were listed in the outline but are not in the slide text I have.

---

## 10. My assignments

### 10.1 A1 — Introduction to LLVM (5 marks, due Aug 10 2025) — OK

- **Setup:** Docker image `llvm20-x86.tar` (LLVM 20.1.7), loaded with `docker load -i` and run with
  `docker run -v host:/root/data -it llvm20`. The tarball is still in `Assignments/Assignment_1/`.
- **Report `CS24S009.pdf`:**
  - sample program with `isCourseAvailableToCredit()` and a `runLoop()` for-loop;
  - IR; CFG PNGs (`cfg_main.png`, ...); call graph;
  - loop info: `Loop at depth 1 containing: %2<header><exiting>,%5,%9<latch>`.
- Nothing to fix.
- Read the IR as a lesson:
  - `-O0` IR uses `alloca`/`load`/`store` with `noinline optnone`.
  - A `for` loop becomes `%2` (cond) → `%5` (body) → `%9` (inc, the latch with `!llvm.loop`) → `%2`, with `%12`
    as the exit.

### 10.2 A2 — Range analysis (15 marks, due Sep 7 2025)

**Spec:**
- flow-sensitive and intraprocedural;
- int locals and arguments only, with + − × arithmetic;
- conditions of the form `var op k`, no nested loops, ≤ 15 lines;
- print `Function f` and then `var : [lo,hi]` at function exit;
- saturating arithmetic;
- widen to FULL at back edges (via `FindFunctionBackedges`);
- implement Kildall's algorithm yourself.

**My implementation** (`Assignment_2/template/Template/analysis/CS24S009.cc`, which equals `rangeAnalysis.cc`, 657
lines):
- `Range` and `BasicBlockState` classes, saturating ops, and `applyCondition`;
- a worklist with back-edge widening.
- It updates ranges **only at stores**. The stored value must be a constant, a load, or a binary operator whose
  operands are loads or constants.
- The only call it handles is `scanf` (address arguments become FULL).

**Verified result: 26/30**, re-run in Oct 2026 with FileCheck exactly as `run_testcases.sh` does. The reference
scores 30/30.

| Test | Program | Expected | Mine | Root cause |
|---|---|---|---|---|
| 17 | `a = abs(a)` (clang emits `llvm.abs.i32`) | a FULL | a [10,10] | Store of a **call** result is ignored |
| 18 | `printf("%x %d", &a, b)` | a FULL | a [10,10] | Alloca passed to a non-`scanf` call not set FULL |
| 19 | `switch(arg){case 10: a=arg; case 20: a=-arg; default: a=30}` | a [−20,30] | a FULL | No **switch-case refinement** (`arg` stays FULL in cases) |
| 21 | `i = (arg>3) ? 30 : 40` (`select`) | i [30,40] | i [10,10] | **`select`** not handled |

- Public tests 1–15 all pass, so private tests 16–30 score 11/15.
- My own `custum_testcases/Readme.txt` also flags failures on while loops (18, 21), complex expressions (25) and
  multiple returns (27).
- **Design lesson.** Track a range for **every SSA value** (the result of every instruction) and let stores just
  copy it, as the reference does. Do not special-case value shapes at the store: nested expressions such as
  `a = b + c*d` then work for free.
- **Phase-2 fix:**
  - evaluate all instructions (binary operator, `select`, `phi`, call → FULL plus alloca arguments → FULL);
  - add switch refinement through `findCaseDest`;
  - keep edge refinement inside the meet;
  - keep sorted-name printing at the unique exit block.
  - Or adopt the reference structure outright, since it is on disk.

### 10.3 A3 — Andersen alias analysis (20 marks, due Oct 2 2025)

**Spec:**
- intraprocedural; `int`, `int*`, `int**`; **globals included**; no `**p = x`;
- variables `a` and `b` always exist;
- print `pts(a) ∩ pts(b)`, e.g. `{ x }`;
- 10 public and 10 hidden tests, 1 mark each.

**My submission** (`analysis/CS24S009.cc`, Oct 2):
- step 1 maps each pointer-typed load to its address operand (`valueToSource`);
- step 2 classifies **stores** into ADDR_OF, COPY, LOAD and STORE;
- then a naive fixpoint loop runs, capped at 100 iterations.

**Verified result: 11/20** (public 10/10, hidden **1/10**: it passes only test 18). The TA's `sol.cc` scores 20/20
in the same harness. Bugs:
1. **Address of a global is treated as COPY.** `store ptr @g, ptr %a` fails the `isa<AllocaInst>(val)` test and
   becomes COPY(a, @g), with pts(@g) = ∅.
   - With globals `a` and `b`, no entry is ever created, so the template's query prints
     `Did you forget to actually implement the pass?`. That happens on tests 11–17, 19 and 20.
2. **`*p = &x` is encoded as STORE(p, x)**, i.e. as `*p = x`, so it propagates pts(x) instead of {x}.
   - It needs a fifth constraint kind: ∀v ∈ pts(p): x ∈ pts(v). The TA solution calls it DEREFERENCE.

**My later rewrite** (`analysis/modifiedImplementation.cc`, which equals `pointerAnalysis.cc`, Dec 5 2025) adds
`isa<GlobalVariable>` handling and correct address-of stores. **Verified 20/20.** A3 needs no further work. Keep
the December version as the canonical one.

### 10.4 A4 — Instrumenting array accesses (20 marks; spec Oct 3, final code Nov 2 2025)

**Spec:**
- extend A2's range analysis to arrays: **treat each array as a single variable, merging all writes**;
- use a **feasibility-aware meet**: if any variable's range is EMPTY, the path is infeasible, so do not merge its
  information;
- in `int test(int,int)`, instrument each array GEP **inline** (no helper calls) so that an out-of-bounds index
  makes the function **return −1**;
- skip the check when the index range is within bounds **or** the site is infeasible, and print a line to
  `errs()` for each skipped access;
- inputs: int locals and 1-D int arrays zero-initialised, indices are variables, no nested loops, ≤ 30 lines,
  `var op k` conditions;
- grading (`run_testcases.sh`):
  - no-instrumentation tests diff the IR (ignoring ModuleID and `; preds`);
  - instrumentation tests run the output under `lli` with three argument pairs.

**My final pass** (`analysis/CS24S009.cc`, which equals `arrayInstrumentation.cc` and `temp4.cc`, Nov 2):
- An RPO sweep repeated 20 times. Per-block maps go from Value to Range, plus `(array, constIndex) → Range` for
  array elements.
- Branch and switch refinement are applied to the merged state.
- An `isInfiniteLoop` heuristic marks the false successor of a loop whose condition variable is never stored as
  unreachable.
- `getRangeAt` replays the block up to the GEP.
- Instrumentation splits the block before the GEP, emits `icmp sge idx, N`, `icmp slt idx, 0` and `or`, branches
  to a per-GEP `sink: ret i32 -1`, and otherwise continues.
- The instrumentation code itself is correct.

**Verified results:**
- Public tests: **10/10**.
- My saved Nov 2 outputs (`testcases/*.ll.out`) also run correctly, and their instrumented-GEP sets equal the TA
  reference outputs in `template.zip` exactly. The reference uses one shared `sink` block and
  `select i1 %a, i1 true, i1 %b`; mine uses one sink per GEP and `or`. Both are fine.
- The Oct 6 draft `temp2.cc` scores 9/10 (fails public test 10).

**Hidden-style tests: 0/5.** I wrote these from the spec, which says *"every control structure except nested loops
has to be handled"*. Every public test with a loop has a *non-terminating* loop, so real loops were untested
publicly.

| Test | Expected | Mine | Root cause |
|---|---|---|---|
| h1 `while (i <= 10) { k = arr[i]; i++; }` (arr[10]) | −1 | 0 (OOB read not caught) | **Back edges are never merged.** `blockExitRanges` is declared *inside* the 20-iteration loop, and the latch is unvisited when the header is processed, so loops are analysed for **one iteration** (i = [0,0]) |
| h2 `while (i < 10) i++; k = arr[i];` | −1 | 0 | Same: the exit edge sees [0,0] ∩ [10,∞) = EMPTY, so the access is skipped as "infeasible" |
| h3 `i = 5; if (i > 10) k = arr[50];` | no instrumentation | instrumented | Infeasibility is detected only if **the index's own** range is EMPTY. The spec wants *any* EMPTY variable (here `i`) to make the whole path infeasible. This is the spec's own example pattern |
| h4 `if (0 ≤ arg1 < 10) idx[arg1] = 25; k = idx[3]; k = arr[k];` | −1 for arg1 = 3 | 0 | Per-index array model: a store with a non-singleton index is **dropped**, and an unknown element reads as [0,0]. The spec's single-variable model gives idx ∈ [0,25] → instrument |
| h5 `if (arg1 < 5) arg1 = -3; if (arg1 < 10) k = arr[arg1];` | −1 for arg1 = 0 | 0 | Edge refinement is applied **after** the meet. The false edge's `arg1 ≥ 5` is intersected with the merged state, which wrongly removes the then-branch value −3 |

Other weaknesses:
- `isInfiniteLoop` misfires on `break` and on any `if (argX > c)` inside a loop. It marks the else successor
  unreachable, and accesses there are skipped unsoundly.
- Blocks the sweep never visits read every load as [0,0], an unsound default.
- Allocas start at [0,0] instead of FULL.

**Phase-2 fix (clean rewrite, about 250 lines):**
1. Start from the TA's A2 reference: Kildall worklist, per-block state, edge-refined meet, back edges met last with
   widening.
2. Model arrays as single variables:
   - `memset` → [0,0];
   - `store v, gep(arr, i)` → `arr := arr ⊔ range(v)` (a weak update);
   - `load gep(arr, i)` → range(arr);
   - GEPs carry no range.
3. Make the meet feasibility-aware: if the edge refinement makes the checked operand EMPTY, **skip that
   predecessor entirely**. A block with no feasible predecessor is unreachable, so its GEPs are skipped and logged.
4. Instrument iff the block is reachable and the index range ⊄ [0, N−1]. Use one shared sink.
5. Regression suite: the 10 public tests, h1–h5 (sources below), and the spec's example program.

**Sources of h1–h5** (each has the standard `main` that prints `test(atoi(argv[1]), atoi(argv[2]))`):
```c
// h1
int test(int arg1, int arg2) { int arr[10] = {0}, i = 0, k = 0;
    while (i <= 10) { k = arr[i]; i = i + 1; } return 0; }          // args (0,0) -> -1
// h2
int test(int arg1, int arg2) { int arr[10] = {0}, i = 0, k = 0;
    while (i < 10) { i = i + 1; } k = arr[i]; return 0; }            // (0,0) -> -1
// h3  (must NOT be instrumented)
int test(int arg1, int arg2) { int arr[10] = {0}, i = 5, j = 50, k = 0;
    if (i > 10) { k = arr[j]; } return 0; }
// h4
int test(int arg1, int arg2) { int arr[10] = {0}, idx[10] = {0}, j = 3, k = 0;
    if (arg1 >= 0) { if (arg1 < 10) { idx[arg1] = 25; } }
    k = idx[j]; k = arr[k]; return 0; }                              // (3,0) -> -1, (5,0) -> 0
// h5
int test(int arg1, int arg2) { int arr[10] = {0}, k = 0;
    if (arg1 < 5) { arg1 = -3; } if (arg1 < 10) { k = arr[arg1]; } return 0; }   // (0,0) -> -1, (7,0) -> 0
```

### 10.5 Lessons across A2–A4

- **Model SSA values uniformly.** Every instruction gets a range or points-to value. Special-casing value shapes at
  stores was the single biggest source of lost marks in both A2 and A3.
- **Transfer on edges, then meet.** This matters both for branch refinement (A4-h5) and for feasibility (A4-h3).
- **Back edges must feed the fixpoint.** The state that carries information around the loop must persist across
  iterations (A4-h1/h2).
- **Public tests are not the spec.** In A2, A3 and A4 the public tests missed exactly the features I left out:
  calls/select/switch, globals, real loops. Write hidden-style tests from every sentence of the spec.

### 10.6 How to re-run everything (no Docker needed)

Arch ships the LLVM runtime libraries (`llvm-libs` 23) but not the headers, `opt` or `lli`. The recipe below avoids
a system install:
```bash
S=<scratch dir>; mkdir -p $S/llvmpkg/root && cd $S/llvmpkg
curl -O https://mirror.osbeck.com/archlinux/extra/os/x86_64/llvm-23.1.1-1-x86_64.pkg.tar.zst   # match `pacman -Q llvm-libs`
curl -O https://mirror.osbeck.com/archlinux/extra/os/x86_64/llvm-23.1.1-1-x86_64.pkg.tar.zst.sig
pacman-key --verify llvm-23.1.1-1-x86_64.pkg.tar.zst.sig llvm-23.1.1-1-x86_64.pkg.tar.zst
tar --zstd -xf llvm-23.1.1-1-x86_64.pkg.tar.zst -C root     # root/usr/{bin/opt,lli,FileCheck; include/llvm}
mkdir -p shim/llvm/Passes && printf '#pragma once\n#include "llvm/Plugins/PassPlugin.h"\n' > shim/llvm/Passes/PassPlugin.h
clang++ -std=c++17 -fPIC -shared -fno-rtti -O1 -Ishim -Iroot/usr/include \
        -include set -include list -include algorithm -include climits  pass.cc -o pass.so
root/usr/bin/opt -load-pass-plugin ./pass.so -passes='range-analysis' -disable-output t.ll 2>&1 | root/usr/bin/FileCheck t.ll
clang -x ir instrumented.ll -o a.out && ./a.out 3 0                  # instead of lli
```
- The LLVM 20 test IR (`#dbg_declare` records) parses fine in LLVM 23, and the passes compile unchanged apart from
  the header shim.
- For a no-instrumentation check, compare against `opt -passes=verify -S` output from the *same* `opt`, so that
  printing differences between versions cancel out.
- For grading-exact confirmation, use the course Docker image (LLVM 20.1.7).

---

## 11. Revision Q&A (past papers; answers verified where marked)

**DFA**
- *RD is monotone, yet it has KILL. How does information increase monotonically?* (MidSem 2015)
  - Monotonicity is about the transfer function f(x) = gen ∪ (x − kill) as a function of its input x.
  - A larger input gives a larger output, because kill is a constant set.
  - Across iterations, each in/out set only grows from {}. Kill removes facts *along a path*, not across
    iterations.
- *Drop `− def(B)` from liveness. Is that the same as flow-insensitive analysis?* **No.**
  - Without kills, v is live at p iff v is used *somewhere downstream* of p.
  - Flow-insensitive analysis says v is live everywhere if it is used anywhere.
  - Counterexample: `B1: y = x; B2: print(y)`. At B2 the modified analysis says x is dead; flow-insensitive says
    x is live.
- *Values of c for `pa(1,2); pa(3,4); pa(5,6)` with `c = a + b`:*
  - context-sensitive: **{3}, {7}, {11}** per call, {3,7,11} overall;
  - interprocedural context-insensitive: a ∈ {1,3,5}, b ∈ {2,4,6}, so **{3,5,7,9,11}**;
  - intraprocedural: **⊤** (any int).
- *Example where adding any one sensitivity changes the result* (MidSem 2017Jan Q5). One program can combine:
  - flow: `x = 1; x = 2;`;
  - path: `if (c) y = 1; else y = 2;` correlated with a later `if (c)`;
  - field: `s.a = 0; s.b = 1`;
  - context: `id(0)` and `id(1)` calls.

**Bit-abstraction questions.** Bits mean "condition definitely true", with conservative 0. Only the bits survive
between statements.
- **MidSem 2015.** AB = (x>2, y<2), starting at 01, non-negative ints:
  - `x=y+2` → 01, because x ∈ [2,3] is not definitely > 2;
  - `x=x+1` → 01;
  - `x=5` → 11;
  - `y=y+2` → 10;
  - `y=x−y` → 10.
  - To answer "is x == 5 at the end?" with YES, track **x == 5** itself (the last write to x is `x = 5`).
- **MidSem 2017Jan.** Bits (y==0, x<2, y<4), unsigned:
  - `y=2` → 001;
  - `x=y/2` → 011, since y<4 gives y/2 ≤ 1;
  - `y=y−x` → 010, since unsigned wrap is possible when x > y;
  - `y=y%1` → 111, since y = 0;
  - `x=++y` → 011, since y = x = 1.
- **MidSem 2016.** Bits (x==0, x>2, y>4), unsigned: **000** after every statement.
  - Each statement's result depends on values that the bits do not retain, e.g. `x = x−y−1` is 0 only if you
    remember x = y+1.
  - The point: a weak abstraction cannot see relations.
- **MidSem 2017.** XY = (p may→null, a may→x), starting at 10:
  - `a=&x` → 11;
  - `x=&y` → 11;
  - `p=*a` → 11, because the bits do not know x → y, so p may be null.

**PTR**
- *Why does collapsing constraint-graph cycles keep precision?* At the fixpoint, nodes on a cycle of subset edges
  have equal sets, so merging them changes nothing (§4.5).
- *Andersen (verified):*
  - MidSem 2014 Q5 → a→{a}, b→{c,q}, c→{a}, q→{a}.
  - MidSem 2016 Q4 (`*p=q, *a=b, c=a, p=&c, d=&a, q=*a, a=&b, *b=p, b=&p, q=&a`) → **a→{b}, b→{p}, c→{a,b,p},
    d→{a}, p→{c}, q→{a,p}**.
  - MidSem 2017 Q5 → **x→{a,c,p,x}**, pts(p) ∪ pts(q) = {a,c,p,x}, pts(a) ∩ pts(q) = {a,c,p,x}. The cycle in the
    final graph runs through a, q, x via load/store edges.
- *Steensgaard (verified):*
  - MidSem 2017 Q7 → {a,c,p,x} → {a,c,p,x}, and q → {a,c,p,x}.
  - MidSem 2016 Q4 → {a,b,c,p} → itself; d, q → {a,b,c,p}.
- *MidSem 2017Jan Q2/Q3 (path-sensitive points-to; Steensgaard).*
  - Flow-insensitive Andersen gives a→{x,n,h6}, b→{n,h6}, c→{h6}, p→{a}, q→{b}.
  - Steensgaard merges {x,n,h6}: a, b and c all → {h6,n,x}; p → {a}; q → {b}.
  - Path-sensitively:
    - on i>0 ∧ i<100: b→h6 and a→h6;
    - on i>0 ∧ i≥100: b→h6 and a→x (via `*p=a`, a self-assignment);
    - on i≤0: a→n.
- *Reverse problem (EndSem 2015 Q2).*
  - Given address-of `a=&x; b=&y; p=&a; q=&b; m=&p` and the final Andersen result, find the load/store
    statements.
  - **Minimum 4 statements** (24 minimal solutions; solver-verified), e.g. **`*a=b; *a=m; *b=p; *m=x`**:
    - x gets {y} and then {p};
    - y gets pts(p);
    - p gets pts(x), which yields y, p → {a,y,p}.
- *Replication for flow-sensitive liveness?*
  - Yes, if the merge is a **join** (union) and each block's in/out is written by a single owner or union-merged.
  - Liveness iterates monotonically from ∅ to the least fixpoint, and chaotic-iteration theorems guarantee the same
    answer.
  - A naive replica that *overwrites* in/out sets can regress facts. The state is per program point, so it is
    larger than for flow-insensitive PTA.

**PAR / concurrency**
- *Relax a barrier soundly.*
  - Let a thread pass early if its post-barrier code does not depend on data written by threads that haven't
    arrived. Variants: split-phase or "fuzzy" barriers (arrive, then independent work, then wait), or point-to-point
    synchronization with only the producers it depends on (stencil neighbours).
- *Lock-free union-find `find`:*
  ```
  find(x): loop { p = parent[x]; if (p == x) return x;
                  gp = parent[p]; if (gp != p) CAS(&parent[x], p, gp);   // path halving, failure is benign
                  x = gp }
  union(x,y): loop { rx=find(x); ry=find(y); if (rx==ry) return;
                     if (rx < ry) swap;  if (CAS(&parent[rx], rx, ry)) return }   // link by index avoids cycles
  ```
- *When is BFS spatially optimal?* When vertices are **numbered in BFS order** (BFS/RCM relabelling, trees laid out
  level by level, grids/paths). Each frontier is then a contiguous ID range and CSR adjacency is read sequentially.
- *Prefix sum for worklist offsets*, A = 5,3,8,2,1,0,2,3, Hillis–Steele with 8 threads:
  - d=1: 5,8,11,10,3,1,2,5
  - d=2: 5,8,16,18,14,11,5,6
  - d=4: 5,8,16,18,19,19,21,24
  - Shift right for the exclusive scan: **B = 0,5,8,16,18,19,19,21**, in log₂8 = 3 steps (Blelloch up/down-sweep
    is the work-efficient alternative).
- *Lock-based pop/process/push vs barrier + atomic version: not equivalent.*
  - With 1 item and 2 threads, the lock version lets T2 pop T1's freshly pushed item.
  - In the barrier version T2 pops from an empty list in phase 1 and pushes `process(garbage)`.
  - Also, `process()` now runs concurrently.

**Design-an-analysis questions.** Write the lattice, direction, meet, transfer functions, initialization, checks,
and assumptions.
- *File opened before read/write.* This is a typestate analysis.
  - Per FILE\* variable: {Unopened, Open, Closed, ⊤}, forward and flow-sensitive.
  - Meet: join to ⊤ when they differ (for a must-check: "definitely open").
  - Transfers: `f = fopen()` → Open; `fclose(f)` → Closed; `fscanf(f)` / `fread(f)` warn unless Open.
  - Aliases need points-to analysis (`g = f`). Correlated branches (`if (c) f = fopen(); …; if (c) fscanf(f)`)
    need path sensitivity.
  - A **single bit** handles only one file with no aliasing and no correlated branches.
- *lock/unlock pairing.* Forward analysis of the **set of possible lock depths** per point (or an interval), with
  union at joins.
  - Errors: depth < 0 (unlock without lock), depth > 0 at exit, or disagreeing depths at a join that later matters.
  - One lock with several unlocks: `lock(); if (c) { unlock(); return; } …; unlock();`.
  - `lock(x)` needs per-lock-variable state plus alias analysis.
  - A barrier is an error if any lock is held. Keep a **held-lock count** (or bit) so that checking a barrier is
    **O(1)**.
- *Dangling pointers.* Points-to analysis plus per-object state {Live, Freed}, forward and flow-sensitive.
  - `free(p)` marks every object in pts(p) Freed (strong update only if pts(p) is a singleton).
  - Dereferencing or freeing a pointer whose targets may be Freed is a warning.
  - Returning `&local` is also dangling (escape check).
- *Out-of-bounds array access.* Range analysis on indices plus sizes (A4), or the polyhedral model for affine loops,
  or annotations.
- *fwrite/fprintf argument mix-up.* A value-flow lattice per value: {⊥, FileHandle, NonFile, ⊤}.
  - `fopen`, `stdout` and `stderr` produce FileHandle; copies propagate; merges join.
  - Error if `fwrite`'s arg 0 may be a FileHandle or its arg 3 is not one; error if `fprintf`'s last argument may
    be a FileHandle while arg 0 is not.
  - It needs points-to analysis for FILE\* stored in memory, and it is interprocedural if handles are passed
    around.
- *Idempotence.* A region is idempotent if re-executing it from its own output state gives the same state.
  - A sufficient syntactic condition: no **clobber antidependence**, i.e. no variable read upward-exposed in the
    region and then written in it (de Kruijf et al., PLDI'12). Under this rule `x = 0` is idempotent and `++x` is
    not.
  - `if (x<0) x = -x` violates the rule yet is semantically idempotent. Prove such cases with abstract
    interpretation: compute the region's effect f on a sign or interval domain and check f(f(s)) = f(s) (after it,
    x ≥ 0, and the region is the identity on x ≥ 0).
  - `time()` and I/O are never idempotent.

---

## 12. Bridges to AI infrastructure (Phase-3 seeds)

1. **Static safety checking for GPU and Triton kernels, especially LLM-generated ones.**
   - A4 combined with range analysis, the polyhedral model and GPU knowledge (CS6023).
   - Kernels from LLM agents (KernelBench-style pipelines) frequently go out of bounds or race. An
     abstract-interpretation checker can prove index bounds per thread block, flag missing masks and
     shared-memory races, and remove redundant bounds checks before expensive benchmarking.
   - It sits close to PTX/MLIR and is publishable. **Strongest candidate.**
2. **Liveness-driven memory planning.**
   - The DFA "memory optimization" lecture is exactly how ML compilers plan activation buffers: live ranges of
     tensors, an interference graph, then offset assignment (TVM storage rewrite, PyTorch Inductor memory planning).
   - The same view applies to LLM KV-cache allocation (paged attention) and to rematerialization/checkpointing
     decisions.
3. **Alias analysis in ML graph compilers.**
   - In-place ops, buffer donation (XLA input–output aliasing), views/strides in PyTorch functionalization. Points-to
     reasoning decides when mutation is safe.
4. **Dependence analysis for fusion and tiling.**
   - ILP/GCD/polyhedral reasoning underlies fusion legality and tiling in MLIR `affine`, Halide/TVM schedules and
     Triton block tiling. RAW/WAR across kernels decides what can be fused or overlapped (streams).
5. **Information-flow (taint) control for LLM agents.**
   - Security analysis plus slicing, and directly continues my CS6570 indirect-prompt-injection project.
   - Label tool outputs as untrusted, propagate labels through the agent's plan and data flow, and block
     sensitive sinks (send email, execute code) on tainted arguments.
   - Related recent work: CaMeL (Debenedetti et al., 2025, "Defeating Prompt Injections by Design") and
     information-flow control for agents (Microsoft FIDES, 2025).
6. **Low-overhead profiling of inference servers.**
   - Ball–Larus-style path IDs and spanning-tree counter placement map onto eBPF/uprobe instrumentation of request
     paths. Sampling vs instrumentation trade-offs carry over directly.
7. **Monotone, unordered computation, as in replication-based PTA.**
   - Join-semilattice merges are CRDTs. Ideas for asynchronous parameter or embedding updates and for graph
     analytics on GPUs (Galois-style worklists, CS6023 L9).

---

## 13. References

- **Textbooks**
  - Aho, Lam, Sethi, Ullman, *Compilers: Principles, Techniques & Tools* (2nd ed.), Ch. 9 (DFA, lattices, constant
    propagation) and Ch. 11 (parallelism, dependence, affine transforms).
  - Khedker, Sanyal, Karkare, *Data Flow Analysis: Theory and Practice* (CRC 2009).
  - Nielson, Nielson, Hankin, *Principles of Program Analysis*.
  - Møller & Schwartzbach, *Static Program Analysis* (free online notes).
  - Allen & Kennedy, *Optimizing Compilers for Modern Architectures*.
- **ExtraMaterial in this folder**
  - M. Lam, CS243 L2–L4 (DFA intro, foundations, constant propagation and speed).
  - `03.pdf` / `04.pdf` (DFA frameworks, worklist algorithm).
  - Collberg, Handout-33 (loop dependence).
  - Aldrich, Hoare logic slides; Dijkstra-style weakest-precondition notes.
  - Warzynski, *Writing an LLVM Pass*.
- **Data-flow foundations**
  - Kildall, "A unified approach to global program optimization", POPL'73.
  - Kam & Ullman, "Monotone data flow analysis frameworks", Acta Informatica 1977.
  - Cousot & Cousot, "Abstract interpretation", POPL'77.
- **Pointer analysis**
  - L. O. Andersen, PhD thesis, DIKU 1994.
  - B. Steensgaard, "Points-to analysis in almost linear time", POPL'96.
  - Hardekopf & Lin, "The Ant and the Grasshopper", PLDI'07 (LCD/HCD).
  - Pereira & Berlin, "Wave propagation and deep propagation for pointer analysis", CGO'09.
  - R. Nasre, "Exploiting the structure of the constraint graph for efficient points-to analysis", ISMM'12.
  - Putta & Nasre, "Parallel replication-based points-to analysis", CC'12.
  - Méndez-Lojo, Mathew, Pingali, "Parallel inclusion-based points-to analysis", OOPSLA'10.
- **Shape analysis**
  - Ghiya & Hendren, "Is it a tree, a DAG, or a cyclic graph?", POPL'96.
  - Sagiv, Reps, Wilhelm, "Parametric shape analysis via 3-valued logic", TOPLAS'02.
- **Slicing**
  - M. Weiser, "Program slicing", ICSE'81 / TSE'84.
  - Ferrante, Ottenstein, Warren, "The program dependence graph", TOPLAS'87.
  - Horwitz, Reps, Binkley, "Interprocedural slicing using dependence graphs", TOPLAS'90.
  - Agrawal & Horgan, "Dynamic program slicing", PLDI'90.
- **Profiling**
  - Ball & Larus, "Optimally profiling and tracing programs", TOPLAS'94.
  - Ball & Larus, "Efficient path profiling", MICRO'96.
  - Ernst et al., Daikon, TSE'01.
- **Security and verification**
  - Wagner, Foster, Brewer, Aiken, BOON, NDSS'00.
  - Foster, Fähndrich, Aiken, CQual (type qualifiers), PLDI'99.
  - Engler et al., metacompilation, OSDI'00.
  - C. A. R. Hoare, "An axiomatic basis for computer programming", CACM 1969.
  - E. W. Dijkstra, *A Discipline of Programming*, 1976.
- **Parallelism**
  - Pingali et al., "The Tao of parallelism in algorithms", PLDI'11.
  - Pugh, "The Omega test", 1991.
  - de Kruijf, Sankaralingam, Jha, "Static analysis and compiler design for idempotent processing", PLDI'12.
