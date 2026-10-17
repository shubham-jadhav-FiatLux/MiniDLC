# Building a Deep-Learning Compiler from Scratch: From Python Tracing to LLVM and CUDA

*A technical retrospective on designing, optimizing, and building MiniDLC — an end-to-end heterogeneous deep learning compiler in C++20 and Python.*

---

## 1. Why Build a Deep Learning Compiler from Scratch?

Every modern ML framework promises effortless execution: write a few lines of Python, invoke `.backward()`, and let the runtime execute on your hardware. But what actually happens between that high-level Python statement and the physical transistors executing fused multiply-accumulate operations on CPU cores or GPU streaming multiprocessors?

Production systems like PyTorch (`torch.compile`), XLA, TVM, and MLIR are millions of lines of code. Their sheer size often obscures the foundational computer science that makes them fast.

**MiniDLC** was conceived as an end-to-end, learning-focused deep learning compiler built strictly from scratch. The ground rule was simple: **no PyTorch imports in the core stack**. From Python operator overloading, to Static Single Assignment (SSA) Graph IR, reverse-mode automatic differentiation, memory planning, loop nest lowering, SIMD vectorization via LLVM, and shared-memory CUDA kernels, every line was designed and written to connect academic coursework directly to production compiler principles.

---

## 2. Architecture: The Two-Level Intermediate Representation

A foundational design decision in MiniDLC is its **two-level IR architecture**:

```
[ Python Frontend ] (x, W, b, loss)
       │
       ▼ Tracing
[ Graph IR (SSA) ] ──► Passes: Shape-Infer, Canonicalize, CSE, DCE, Autodiff, Fusion, Mem-Plan
       │
       ▼ Lowering
[ Loop IR (Nests) ] ──► Passes: Tiling, Unrolling, Vectorization, Parallelization, Bounds-Check Elimination
       │
       ├─► [ LLVM CPU Backend ] ──► AVX2 Vector SIMD + ThreadPool JIT
       ├─► [ CUDA GPU Backend ] ──► Shared-Memory Tiled CUDA + NVRTC
       └─► [ NumPy Reference ]  ──► Float64 Precision Ground Truth
```

### Why Two Levels?
High-level tensor graphs excel at whole-program semantic transformations: algebraic rewriting ($x + 0 \to x$), dead-code elimination, automatic differentiation, and memory reuse planning. However, tensor graphs are ill-suited for hardware-specific loop scheduling. 

Conversely, low-level loop nests excel at expressing iteration spaces, memory strides, cache blocking, and vector register packing, but are terrible for automatic differentiation. MiniDLC bridges this by transforming SSA Graph IR into Loop IR through declarative lowering templates.

---

## 3. Reverse-Mode Automatic Differentiation (Autodiff)

In MiniDLC, backpropagation is not an eager tape; it is a **graph-to-graph compilation pass**. Given a scalar loss node $L$, the `AutodiffPass`:
1. Traverses the forward graph in reverse topological order.
2. Seeds the loss gradient $dL = 1.0$.
3. Evaluates per-op Vector-Jacobian Product (VJP) rules to construct backward nodes.
4. **Adjoint Accumulation**: When a value fans out to multiple consumers, its incoming adjoint contributions are summed (`add` operations appended to the graph).
5. Appends parameter updates (e.g. Adam or SGD) directly into the graph.

Because the backward pass and optimizer updates are appended to the same graph, **one compiled graph equals one complete training iteration**. Every backward gradient VJP was verified against double-precision (`float64`) central finite differences ($\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$) to within $10^{-3}$ tolerance.

---

## 4. Compiler Passes That Actually Matter

### A. Epilogue Fusion ($1.39\times$ DRAM Memory Traffic Reduction)
In a naive deep learning framework, an MLP layer `relu(x @ W + b)` generates three separate kernel invocations:
1. Matrix multiplication: writes $C_{\text{mm}}$ to DRAM ($131.1\text{ KB}$).
2. Bias addition: reads $C_{\text{mm}}$ and $\text{bias}$, writes $C_{\text{bias}}$ to DRAM ($131.1\text{ KB}$).
3. ReLU activation: reads $C_{\text{bias}}$, writes $C_{\text{relu}}$ to DRAM ($131.1\text{ KB}$).

MiniDLC's `FusionPass` identifies producer-consumer relationships and folds bias addition and ReLU directly into the matrix multiplication accumulation epilogue. The dot product remains resident in hardware registers and L1 cache:
$$\text{Total DRAM Bytes Moved}: 1,860.6\text{ KB} \longrightarrow 1,336.3\text{ KB} \quad (1.39\times \text{ reduction})$$
This yields a measured **$2.63\times – 2.77\times$ speedup** on CPU over the unfused sequence.

### B. Liveness-Based Memory Planning ($26.2\%$ Memory Saved)
Rather than invoking runtime dynamic memory allocators (`malloc`/`cudaMalloc`) per tensor, MiniDLC statically plans memory across a single contiguous arena:
- Computes liveness intervals $[t_{\text{start}}, t_{\text{end}}]$ over the execution schedule.
- Constructs an interference graph where edges represent overlapping lifetimes.
- Solves greedy interval coloring (largest-first).
- Intermediate forward activations that are consumed before backpropagation completes yield their arena slots to backward gradients.

This reduced peak memory footprint for the MNIST training graph from **$1.64\text{ MB}$** to **$1.21\text{ MB}$** (**$26.2\%$ memory reduction**).

### C. Static Range-Analysis Bounds-Check Elimination & Loop Versioning (`CS6843 A2/A4`)
Vectorized loops require memory bounds guards to prevent out-of-bounds page faults. MiniDLC implements formal static interval range analysis:
- Loop variables are mapped to intervals $\text{Env}: v \mapsto [0, \text{bound}-1]$.
- Affine indexing expressions $A[\text{idx}]$ are evaluated interval-wise.
- When tensor dimensions are exact multiples of vector widths and tile sizes (e.g. $128 \times 256$ with $W=8$), static analysis proves $0 \le \text{idx} < \text{extent}(A)$, completely stripping runtime bounds guards (`attributes {bounds_check_eliminated = true}`).
- When dimensions are non-multiples (e.g. $130$), the pass applies **loop versioning/peeling**: splitting the loop into an unguarded vectorized main loop ($0..128$) and a scalar cleanup tail loop ($128..130$).

---

## 5. Code Generation: From Loop IR to Silicon

### LLVM CPU Backend (AVX2 SIMD & ThreadPool)
The CPU backend traverses the optimized `KernelNest` and emits native LLVM IR:
- Maps innermost loops to `<8 x float>` vector registers.
- Replaces separate multiplications and additions with hardware Fused Multiply-Add intrinsics (`llvm.fmuladd`).
- Replaces naive max with `llvm.maximum` to preserve IEEE-754 / PyTorch NaN-propagation semantics (`relu(NaN) == NaN`).
- Partitions outer parallel loops across a persistent C++ `ThreadPool`, executing across physical CPU cores without OpenMP runtime overhead.

### CUDA GPU Backend (Shared Memory & Warp Shuffles)
Targeting NVIDIA GPUs (CS6023), the CUDA backend emits optimized CUDA C++:
- **Shared-Memory 2D Tiling ($16 \times 16$)**: Stages operand tiles into `__shared__ float sA[16][16]` and `sB[16][16]`, delivering $16\times$ data reuse and eliminating $16\times$ DRAM bandwidth demands.
- **On-Chip Transposition**: Matrix transpositions during backward passes ($X^T dY$ and $dY W^T$) are performed in shared memory, maintaining 100% coalesced 32-byte/64-byte global memory transactions.
- **Warp-Shuffle Reductions (`__shfl_down_sync`)**: Row-wise softmax and cross-entropy evaluate in-warp reduction trees across 5 butterfly steps ($16 \to 8 \to 4 \to 2 \to 1$) without shared memory allocations or barrier stalls.
- **Fused Multi-Tensor Adam**: Updates parameters, gradients, and moments in registers within a single pass, saving up to $66\%$ of optimizer memory traffic.

---

## 6. Empirical Performance Results

All numbers reported below were empirically measured on the host platform:

### 1. Hardware Baseline & Roofline Characterization
- **Single-Core AVX2 Peak**: $112.5\text{ GFLOP/s}$ (dense FMA benchmark)
- **Multi-Core AVX2 Peak**: $206.0\text{ GFLOP/s}$
- **DRAM STREAM Copy Bandwidth**: $14.09\text{ GB/s}$
- **Hardware Ridge Point**: $I^* = 7.98\text{ FLOP/byte}$

### 2. End-to-End MNIST Training
- **Model**: MLP ($784 \to 256 \to 10$), batch size $128$, Adam optimizer.
- **Reference Interpreter (NumPy)**: $77.2\text{ s}$ for 2 epochs ($96.93\%$ test accuracy).
- **MiniDLC Native LLVM CPU Backend**: **$34.0\text{ s}$** for 2 epochs (**$96.88\%$ test accuracy**).
- **Speedup**: **$2.27\times$ wall-clock speedup** over the reference implementation.

### 3. Autoregressive Character-Level Transformer (Char-GPT)
- **Model**: 30,656 trainable parameters, 2 layers, 2 heads, $d=32$, context 32.
- **Dataset**: TinyShakespeare (1,115,394 characters, 65-character vocabulary).
- **Training Time**: 40 iterations completed in **$0.78\text{ s}$** via `dl.compile`.
- **Convergence**: Empirical cross-entropy loss dropped from **$4.3277$** to **$3.1475$**, correctly learning character transition statistics and sampling readable text.

---

## 7. Reflections & Conclusion

Building MiniDLC reinforced three enduring systems lessons:
1. **Memory traffic dominates compute**: Even highly optimized vector SIMD instructions sit idle if data movement is not minimized. Fusion and liveness-based arena reuse deliver far greater wall-clock speedups than micro-optimizing ALU arithmetic.
2. **IR design governs optimization power**: Keeping Graph IR and Loop IR separated prevents abstraction leaks and allows each representation to do what it does best.
3. **Rigorous testing is non-negotiable**: Floating-point gradient checks, golden FileCheck lit tests, and reference differential tests caught subtle bugs (such as fan-out adjoint accumulation and non-tile remainders) that would have silently degraded model convergence.

MiniDLC demonstrates that with principled systems programming, solid dataflow algorithms, and hardware-conscious engineering, building a complete, functioning deep-learning compiler from scratch is not only achievable—it is one of the most rewarding endeavors in computer science.
