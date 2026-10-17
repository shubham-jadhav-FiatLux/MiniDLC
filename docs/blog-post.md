# Building a Deep-Learning Compiler from Scratch: What Worked, What Didn't, and the Redesign

*Draft. A retrospective on MiniDLC v1 and on the audit that reshaped it into v2.*

---

## 1. Why build one

Frameworks make training look effortless: write a model, call `.backward()`, and the runtime does the rest. Production
compilers such as XLA, TVM and `torch.compile` are large enough that the computer science inside them is hard to see.
MiniDLC is a learning project that builds the whole stack once — Python frontend, SSA graph IR, autodiff, optimization
passes, a loop-nest IR, code generation through LLVM, a runtime — with one rule: no PyTorch in the core.

## 2. Two levels of IR

A tensor graph is the right place for whole-program rewrites: algebraic simplification, dead-code elimination, automatic
differentiation, deciding which ops share a kernel, planning memory. It is the wrong place to talk about cache blocking,
vector registers or GPU threads. A loop-nest IR is the opposite. MiniDLC therefore lowers each group of fused graph ops
into a loop nest, and code generation works from the loops.

## 3. What worked

- **Autodiff in the reference interpreter.** The NumPy interpreter implements reverse-mode autodiff for 18 ops,
  including batched matmul, numerically stable cross-entropy, LayerNorm and an embedding scatter-add. Every VJP is
  checked against float64 central finite differences, a seeded fuzzer differentiates random graphs, and contract tests
  pin down NaN/Inf behaviour and overflow. It trains an MNIST MLP to 96.93% and a small character-level Transformer to
  2.10 nats held-out cross-entropy on TinyShakespeare, below the 2.45-nat bigram baseline.
- **LLVM IR by hand.** The MNIST training step runs on AVX2 kernels written directly in LLVM IR: `<8 x float>` vectors,
  `llvm.fmuladd`, and `llvm.maximum` so that `relu(NaN)` stays NaN, as in PyTorch. Compiled at run time with Clang and
  called through ctypes, they reach 96.88% test accuracy.
- **Compiler infrastructure.** A C++ SSA graph IR with a builder, printer, parser and verifier; graph passes; a loop IR
  with interval range analysis that proves accesses in bounds and peels vector remainders; and `dlc-opt`, which prints
  the IR after every pass.

## 4. What the audit found

v1 was marked complete after its six milestones. A systematic review — a sanitizer build, small programs aimed at
each pass, re-measurement — showed otherwise.

- **The two halves never met.** The Python frontend and the C++ compiler did not talk to each other.
  `dl.compile(target="cpu")` ran the hand-written kernels for a fixed two-layer ReLU MLP whatever the model was: given a
  GELU network, a network with scaled logits, or a three-layer network, it silently computed the fixed MLP.
- **Code generation by template.** The loop IR could not say which GPU thread runs a loop, which memory a buffer lives
  in, where a barrier goes, or where a bounds check is needed. So the backends recognised kernels by name and pasted
  templates; a division kernel came out as an addition.
- **Tests that could not fail.** The unit tests were built in Release mode, where `assert` compiles to nothing. Most
  tests checked that a pass reported a change, not what it produced; the autodiff golden test checked only forward ops,
  so it missed that the C++ ReLU gradient multiplied by the input instead of masking it.
- **The wrong baselines.** The 2.27× "speedup over the interpreter" and the 2.77× "fusion speedup" were measured against
  a NumPy linked to the netlib reference BLAS. With a standard NumPy the interpreter is 2.3× faster than the hand-written
  kernels, and fusing the epilogue into the same GEMM changes nothing measurable at this size (1.00×).
- **Memory planning at the wrong level.** Liveness was computed op by op, but a fused kernel runs several ops at once, so
  its output was placed inside the weight matrix it was still reading. Running the C++ pipeline end to end on the
  simplest fused layer gave errors of 4.7×10⁷.

## 5. Lessons

1. **Connect the pipeline end to end first,** even with trivial passes. A compiler whose output never runs cannot be
   validated, and every later number inherits that.
2. **A test that cannot fail is not a test.** Assertions must survive Release builds, tests must check outcomes, and
   known bugs belong in the test suite — as disabled tests and expected failures — rather than in prose.
3. **The IR must express every decision the backend needs.** Otherwise the backend makes those decisions itself, by
   pattern matching.
4. **Compare like with like, against standard baselines.** A fusion ablation must use the same kernel with and without
   fusion; a speedup must be measured against a NumPy with an optimized BLAS.
5. **An analysis matters only when a transformation uses it.** Bounds-check elimination without bounds checks is an
   attribute in a printout.

## 6. The v2 design

- **The textual graph IR is the contract** between the Python frontend and the C++ compiler. Because the reference
  interpreter executes the same text, every pass can be validated by running the graph before and after it.
- **A loop IR that can express GPUs:** loops bound to blocks and threads, buffers in global, shared or register memory,
  barriers, warp shuffles, atomics, and explicit guards for range analysis to delete. Each fused kernel gets a per-target
  schedule; no backend decides anything on its own.
- **One LLVM emitter, two targets:** x86-64 through ORC JIT, and NVIDIA GPUs through LLVM's NVPTX backend, which emits
  PTX that the CUDA driver compiles at load time. PTX needs no GPU or CUDA toolkit to produce — Clang 23 on the
  development laptop already turns a hand-written NVPTX kernel into `shfl.sync.down`, `bar.sync` and `max.NaN.f32`
  instructions — so GPU code is generated and golden-tested locally and run on a Colab T4.
- **Phases,** each ending in a tested state: truthful docs and real tests (P0, done) → IR and Python↔C++ bridge (P1) →
  training graph in C++ (P2) → fusion and buffer assignment (P3) → CPU backend (P4) → GPU backend (P5) → Transformer
  on both targets (P6) → measurements and write-up (P7).

## 7. Where things stand

The tests now fail when they should: 21 unit tests with assertions in every build, 13 known bugs recorded as disabled
tests, lit tests checked by LLVM's FileCheck, gradient checks for all 18 ops, and CI with AddressSanitizer and
UndefinedBehaviorSanitizer. The next step is the bridge between the frontend and the compiler.
