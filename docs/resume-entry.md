# MiniDLC — Resume Entry

Rules: list only tools actually used; every number comes from [`perf-report.md`](perf-report.md), and every number there
from a script in this repository; the entry describes what exists, not what is planned. The 2026-10-17 audit withdrew
the previous entry, which claimed ORC JIT, CUDA training, and fusion and memory-planning numbers that the code did not
support.

## Current entry (accurate as of P0)

```latex
\rowcolor{SecondaryColor}
\CellStrut{\EntryTitleFont MiniDLC: Deep-Learning Compiler Stack} \textit{| C++20, LLVM IR, Python} \hfill {\DateRange{Oct'26}{Present}} \\
\vspace{-1.5ex}
\begin{Bullets}
  \item Built a deep-learning compiler stack in \textbf{C++20/Python}: a tracing frontend, an \textbf{SSA graph IR} with an opt-style pass driver (shape inference, CSE, DCE, fusion, liveness-based memory planning), and a \textbf{loop-nest IR} with interval \textbf{range analysis}.
  \item Implemented \textbf{reverse-mode autodiff} for 18 tensor ops, verified against \textbf{float64 finite differences} and a 25-seed random-DAG fuzzer; trained a 30.7K-parameter \textbf{character-level Transformer} to 2.10 nats held-out loss (bigram baseline: 2.45).
  \item Hand-wrote \textbf{AVX2 LLVM IR} training kernels (fused matmul+bias+ReLU, Adam) compiled at run time with \textbf{Clang}, training an \textbf{MNIST MLP} to 96.9\% test accuracy; set up CI with \textbf{GoogleTest}, \textbf{lit/FileCheck} and \textbf{ASan/UBSan}.
\end{Bullets}
\vspace{-1ex} \\ \hline
```

**Swap-ins for different roles** (same evidence):
- *Compiler / program-analysis roles, bullet 1:* `\item Built an \textbf{SSA graph IR} and opt-style pass driver in \textbf{C++20} — shape inference by abstract interpretation, value-numbering CSE, reachability DCE, fusion, liveness-based memory planning — and a \textbf{loop-nest IR} with interval \textbf{range analysis} and remainder-loop peeling.`
- *Performance roles, bullet 3:* `\item Hand-wrote \textbf{AVX2 LLVM IR} training kernels (fused matmul+bias+ReLU, Adam) compiled at run time with \textbf{Clang}, training an \textbf{MNIST MLP} to 96.9\% test accuracy; profiling against a measured FMA/STREAM \textbf{roofline} traced 79\% of step time to the scalar backward GEMM.`

| Claim | Evidence |
|---|---|
| Tracing frontend, SSA graph IR, opt-style pass driver | `python/minidlc/`, `include/dlc/ir/`, `lib/passes/`, `tools/dlc-opt/` (status: `architecture.md`, Part 2) |
| Loop-nest IR with interval range analysis, remainder-loop peeling | `lib/loopir/LoopPasses.cpp`; `tests/lit/bounds_check_elim.dlc`, `tests/lit/loop_versioning.dlc` |
| Reverse-mode autodiff for 18 ops, finite-difference checks, 25-seed fuzzer | `python/minidlc/reference.py`; `tests/python/test_gradcheck.py`, `tests/python/test_fuzzer.py`; perf report §8 |
| 30.7K-parameter Transformer, 2.10 nats held-out (bigram 2.45) | `benchmarks/chargpt_heldout.py`; perf report §7 |
| Hand-written AVX2 LLVM IR kernels compiled with Clang | `python/minidlc/llvm_engine.py` |
| MNIST MLP, 96.9% test accuracy | `examples/mnist_mlp.py`; perf report §5 |
| Roofline profiling, 79% of step time in the backward GEMM | `benchmarks/roofline.py`, `examples/mnist_mlp.py --profile`; perf report §1, §5 |
| CI with GoogleTest, lit/FileCheck, ASan/UBSan | `.github/workflows/ci.yml`, `tests/unit/`, `tests/lit/` |

**Not to claim yet:** ORC JIT; CUDA, NVRTC, PTX or any GPU execution; "end-to-end" compilation of models to native
code; multi-threaded execution; fusion speedups or memory savings; multi-head attention. Each becomes claimable in the
phase listed in `architecture.md`.

## Target entry (after P7)

Fill the bracketed numbers from the perf report once they are measured.

```latex
\begin{Bullets}
  \item Engineered an end-to-end \textbf{deep-learning compiler} lowering models from a custom \textbf{Python} frontend through C++ \textbf{SSA graph} and \textbf{loop IRs} to x86-64 and \textbf{NVIDIA PTX} via \textbf{LLVM}.
  \item Implemented \textbf{reverse-mode autodiff} with activity analysis, dependence-checked \textbf{operator fusion}, and liveness-based \textbf{memory planning} cutting peak memory by [X]\% on the training graph.
  \item Generated vectorized CPU code via \textbf{ORC JIT} and tiled GPU kernels via the \textbf{NVPTX} backend (shared memory, warp shuffles), training an \textbf{MNIST MLP} and \textbf{char-GPT} on CPU and a T4 GPU at [Y]\% of roofline.
\end{Bullets}
```
