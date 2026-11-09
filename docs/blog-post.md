# Building a Deep-Learning Compiler from Scratch, Twice

*Draft for publication. A retrospective on MiniDLC: a v1 that looked finished and was not, the audit that showed it,
and the v2 that compiles a whole training step — forward, backward and optimizer — to native CPU code and to PTX.
Every number here comes from a script in the repository; [`docs/perf-report.md`](perf-report.md) has the details.*

---

## 1. Why build one

Frameworks make training look effortless: write a model, call `.backward()`, and the runtime does the rest. Production
compilers such as XLA, TVM and `torch.compile` are large enough that the computer science inside them is hard to see.
MiniDLC builds the whole stack once — a Python frontend, an SSA graph IR, automatic differentiation, optimization
passes, a loop-nest IR, code generation through LLVM for x86-64 and NVIDIA GPUs, a runtime — with one rule: no
PyTorch in the core. PyTorch appears only as the reference the results are checked against.

## 2. The first version, and the audit

v1 had all the boxes ticked: a tracing frontend, a C++ graph IR with passes, a loop IR with range analysis, LLVM and
CUDA backends, a trained MNIST MLP and a small Transformer. A systematic audit — a sanitizer build, small programs
aimed at each pass, re-measurement — found that the pieces did not add up:

- **The two halves never met.** `dl.compile(target="cpu")` ran hand-written kernels for a fixed two-layer MLP whatever
  the model was: given a GELU network or a three-layer one, it silently computed the fixed MLP.
- **Code generation by template.** The loop IR could not say which GPU thread runs a loop, which memory a buffer lives
  in, or where a barrier goes, so the backends recognised kernels by name and pasted templates.
- **Tests that could not fail.** Release builds compiled every `assert` away; most tests checked that a pass reported a
  change, not what it produced. The C++ ReLU gradient multiplied by the input instead of masking it, unnoticed.
- **The wrong baselines.** A claimed 2.77× fusion speedup had been measured against a NumPy linked to the netlib
  reference BLAS; against a standard NumPy, fusing the epilogue into the same GEMM changed nothing measurable.
- **Memory planning at the wrong level.** Liveness op by op placed a fused kernel's output inside the weight matrix it
  was still reading: errors of 4.7×10⁷ on the simplest fused layer.

The lessons became the rules for v2: connect the pipeline end to end first; a test that cannot fail is not a test; the
IR must express every decision a backend needs; compare like with like; an analysis matters only when a
transformation uses it.

## 3. The v2 pipeline

- **A textual graph IR is the contract** between Python and C++. The NumPy reference interpreter executes the same
  text, so every graph pass is checked by running the graph before and after it (translation validation).
- **The training step is one graph.** Reverse-mode autodiff with activity analysis (no gradient for the input batch:
  half the backward FLOPs of the MNIST step), then the Adam or SGD update appended as ops whose results take over the
  parameters' buffers.
- **Fusion and buffer assignment.** Elementwise ops merge; a matmul absorbs its epilogue — bias, ReLU, even the Adam
  update of the weight it computes the gradient of. Buffer assignment works per fused kernel, keeps parameters and
  optimizer state in a persistent region, updates them in place when that is provably safe, and a checker proves that
  no two live buffers overlap.
- **A loop IR that can express both targets.** Loops are serial, parallel, vectorized, unrolled, or bound to GPU blocks
  and threads; buffers live in global, shared or register memory; barriers, warp shuffles, atomics and guards are
  statements. Per-target schedules decide everything; the LLVM emitter only translates.
- **Loop passes.** Range analysis proves accesses in bounds and deletes guards; a dependence check verifies every
  parallel and vectorized loop; a simplifier propagates constants and hoists invariants. On the GPU, a warp probe
  measures bank conflicts and coalescing, a padding pass removes the conflicts, partial tiles are versioned so that
  interior blocks run without guards, and a resource check rejects launches the hardware could not run.
- **One emitter, two targets.** x86-64 through LLVM's O3 pipeline and ORC LLJIT, with compiled code cached on disk; NVIDIA
  GPUs through LLVM's NVPTX backend, which produces PTX without a GPU or a CUDA toolkit.

## 4. Results on the CPU

On a 4-core laptop (i7-8550U), the compiled MNIST training step (784-256-10, batch 128, Adam) takes **1.88 ms**,
against 2.23 ms for PyTorch on the CPU and 6.1 ms for the NumPy interpreter; two epochs reach 96.9% test accuracy.
Ablations of the same compiler: without fusion the step takes 1.6× longer, with run-time checks instead of
range-analysis proofs 2.1×, and with untiled matmuls 22×. The first-layer GEMM reaches 56 GFLOP/s, about half of
OpenBLAS (102): a single 4×16 register-tile schedule is not a tuned BLAS, and the report says so.

The character-level Transformer (2 blocks, d_model 64, 4 heads) trains through the generated code to **1.93 nats**
held-out on TinyShakespeare after 2,000 steps — PyTorch from the same weights on the same batches: 1.93; a bigram
model: 2.48 — with a step of 11.1 ms against PyTorch's 14.3 ms. In float64, the reference's loss and every gradient
match PyTorch's to 1e-7.

## 5. The GPU without a GPU

The development laptop has no NVIDIA GPU, so the GPU path was built to be checked without one:

- **A lockstep interpreter** runs the GPU loop IR with SIMT semantics: branches split the active threads, a barrier
  reached by part of a block is an error, shuffles must read active lanes, shared memory starts as NaN. A second audit
  found the gap in this design: executing a block in lockstep hides a missing barrier, since every statement finishes
  for all threads before the next. The interpreter now records every access and reports any two threads that touch an
  element without a barrier between them, one of them writing — in shared memory between barriers, in global memory
  across the whole launch.
- **ptxas** assembles every kernel: the Char-GPT training step is 316 kernels with no register spills.
- **Artifacts** — the plan as JSON plus the PTX — run on any NVIDIA GPU through a 400-line ctypes runtime, so a Colab T4
  needs neither LLVM nor the compiler. The script checks the GPU's losses against the reference, trains, evaluates,
  times kernels with CUDA events, and compares with PyTorch and cuBLAS on the same GPU.

The bank-conflict pass is a small example of why analyses should drive transformations. The hand-written schedule
padded the matmul's transposed shared tile from 64 to 65 words, the textbook fix. The pass measured each candidate
pitch with the warp probe and chose 66: with 65, a warp's stores still collide two to a bank.

## 6. How it is checked

Translation validation of every graph pass; every VJP against float64 finite differences; every op through the
generated code on every target against the reference, with odd shapes, NaN and Inf; a fuzzer that builds random
models and trains them through the whole compiler against eager training in the reference; end-to-end tests nightly;
AddressSanitizer and UndefinedBehaviorSanitizer in CI. Checkpoints written by one target resume on any other, which
tests the runtime's state handling across all of them.

## 7. What is left

The run on a real GPU: until `examples/colab_gpu.py` has run on a T4, the GPU path is verified only by simulation and by
ptxas, and the GPU numbers do not exist. Beyond that, the stretch goals: a CUDA-C emitter as a second GPU path,
tensor cores through inline PTX, an autotuner for tile sizes.
