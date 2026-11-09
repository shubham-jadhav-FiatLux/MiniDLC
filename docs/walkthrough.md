# Walkthrough: one training step through the compiler

This follows a deliberately tiny model, a linear layer with ReLU and a cross-entropy loss, trained with SGD, through
every stage of MiniDLC. Every listing below is real output; the commands reproduce it. The same path compiles the MNIST
MLP and Char-GPT.

```python
import minidlc as dl
from minidlc import nn, optim

lin = nn.Linear(3, 2)
x = dl.placeholder((4, 3), name="x")
y = dl.placeholder((4,), dtype="i32", name="y")
loss = dl.cross_entropy(dl.relu(lin(x)), y)
step = dl.compile(loss, optimizer=optim.SGD(lin.parameters(), lr=0.1), target="cpu")
step(x=xb, y=yb)   # forward, backward and the update; returns the loss
```

## 1. Tracing and the graph IR

Operations on `Tensor`s build a graph; `minidlc.ir.export` prints it as the textual graph IR, the contract between
Python and the C++ compiler (`docs/ir-spec.md`). Parameters are arguments marked `{param}`; the shapes are static.

```
func @step(%x: f32[4,3], %y: i32[4], %weight: f32[3,2] {param}, %bias: f32[2] {param}) -> f32[] {
  %0 = matmul(%x, %weight) : f32[4,2]
  %1 = add(%0, %bias) : f32[4,2]
  %2 = relu(%1) : f32[4,2]
  %3 = cross_entropy(%2, %y) : f32[]
  return %3
}
```

`dl.compile` hands this text to the compiled module (`_minidlc.compile`); `dlc-opt` runs the same passes from the
command line.

## 2. Autodiff and the optimizer, in the graph

```bash
./build/dlc-opt input.dlc "--passes=autodiff,sgd{lr=0.1},canonicalize,cse,dce"
```

Activity analysis finds the values that depend on a parameter and reach the loss; reverse-mode autodiff appends their
vector-Jacobian products (`lib/passes/Autodiff.cpp`); optimizer lowering appends the update and marks the new
parameters as donated results — they replace the arguments after the step (`lib/passes/Optimizer.cpp`).
Canonicalization folds the transpose of `%x` into the matmul's `transpose_a` flag.

```
func @step(...) -> (f32[], f32[3,2] {donates = %weight}, f32[2] {donates = %bias}) {
  %0 = matmul(%x, %weight) : f32[4,2]
  %1 = add(%bias, %0) : f32[4,2]
  %2 = relu(%1) : f32[4,2]
  %3 = cross_entropy(%2, %y) : f32[]
  %4 = onehot(%y) : f32[4,2] {num_classes = 2}
  %5 = softmax(%2) : f32[4,2] {axis = -1}
  %6 = sub(%5, %4) : f32[4,2]                        // d loss / d logits = (softmax - onehot) / N
  %8 = mul(%6, %7) : f32[4,2]                        // %7 = 0.25
  %10 = cmp_gt(%1, %9) : f32[4,2]                    // ReLU's mask
  %11 = select(%10, %8, %9) : f32[4,2]
  %12 = sum(%11) : f32[1,2] {axes = [0], keepdims = 1}
  %13 = reshape(%12) : f32[2] {shape = [2]}          // d loss / d bias
  %14 = matmul(%x, %11) : f32[3,2] {transpose_a = 1} // d loss / d weight = x^T g
  %16 = mul(%14, %15) : f32[3,2]                     // %15 = 0.1, the learning rate
  %17 = sub(%weight, %16) : f32[3,2]
  %18 = mul(%13, %15) : f32[2]
  %19 = sub(%bias, %18) : f32[2]
  return %3, %17, %19
}
```

Every graph pass is translation-validated in the tests: the transformed function must compute what the original
did, in the reference interpreter.

## 3. Fusion and the cost model

Fusion (`lib/passes/Fusion.cpp`) groups the ops into kernels: elementwise ops of one shape merge, and a matmul
absorbs elementwise consumers as its epilogue — here the bias, ReLU and ReLU mask go into the forward matmul, and the
SGD update into the weight-gradient matmul. A merge is legal only if the graph of kernels stays acyclic and nothing in
the epilogue feeds the matmul.

```bash
./build/dlc-opt step.dlc --print-cost
```
```
  kernel                                   kind                  FLOPs          bytes  FLOP/byte
  k0_matmul_add_relu_cmp_gt                matmul                   72            144       0.50
  k1_cross_entropy                         row                      40             52       0.77
  k2_onehot                                gather                    0             48       0.00
  k3_softmax                               row                      40             64       0.62
  k4_sub_mul_select                        elementwise              24            128       0.19
  k5_sum                                   reduction                 8             40       0.20
  k6_matmul_mul_sub                        matmul                   60            128       0.47
  k7_mul_sub                               elementwise               4             24       0.17
  total: 248 FLOPs; 628 bytes fused vs 1048 bytes op by op (40.1% less)
```

## 4. Buffer assignment

```bash
./build/dlc-opt step.dlc --print-mem-report
```
```
  persistent region: 512 bytes (params, optimizer state, consts)
  transient arena:   1536 bytes, vs 2304 without reuse (33.3% less)
  in-place updates:  2, copied at the end: 0
  %weight                       persistent offset          0  size         24  live -1..8
  %bias                         persistent offset        256  size          8  live -1..8
  %2                            transient  offset        256  size         32  live 0..3
  %11                           transient  offset        256  size         32  live 4..6
  ...
```

Parameters and optimizer state live in a persistent region; everything else gets an offset in a transient arena by
interval colouring over live ranges measured in kernels (`%2` and `%11` share bytes because their lives don't
overlap). Both updates are written in place: the kernel computing new weights reads `%weight` only at the element it
writes, and after every other reader. A checker verifies every assignment.

## 5. Loop IR with CPU schedules

```bash
./build/dlc-opt step.dlc --emit-loopir
```

Each kernel class has a schedule (`lib/loopir/Lowering.cpp`). The weight-gradient matmul is a register tile (here the
whole 3×2 output) accumulated over k, with the SGD update applied to the tile in registers and stored over `%weight`:

```
kernel @k6_matmul_mul_sub(%x: f32[4,3], %11: f32[4,2], %weight: f32[3,2]) {
  alloc acc0: f32[3,2] register
  for p in 0..1 {
    for ii1 in 0..3 unroll { for jj2 in 0..2 unroll { acc0[((ii1 * 2) + jj2)] = 0.0; } }
    for k3 in 0..4 {
      for ii1 in 0..3 unroll {
        for jj2 in 0..2 unroll {
          acc0[((ii1 * 2) + jj2)] = fma(%x[((k3 * 3) + ((p * 4) + ii1))], %11[((k3 * 2) + jj2)], acc0[…]);
        }
      }
    }
    for ii1 in 0..3 unroll {
      for jj2 in 0..2 unroll {
        f32 acc4 = acc0[((ii1 * 2) + jj2)];
        f32 v5 = (acc4 * 0.1);
        f32 in6 = %weight[((((p * 4) + ii1) * 2) + jj2)];
        f32 v7 = (in6 - v5);
        %weight[((((p * 4) + ii1) * 2) + jj2)] = v7;
      }
    }
  }
}
```

On the MNIST layers the same schedule uses 4×16 tiles and a parallel outer loop. Then the loop passes run
(`lib/loopir/Analysis.cpp`): range analysis proves every access above in bounds (`--print-loop-report`), indices
read from memory (embedding ids, labels) get explicit guards, and a dependence check confirms that parallel iterations
write disjoint elements and vectorized loops carry no dependence. Last, simplification (`lib/loopir/Transforms.cpp`)
propagates integer constants, folds divisions that the loop ranges decide, and hoists loop-invariant `let`s; this tiny
kernel gives it nothing to do.

## 6. LLVM and the JIT

The emitter (`lib/backends/llvm/Emitter.cpp`) translates the loop IR mechanically: `noalias` buffer pointers,
unrolled loops expanded into SSA values, `vectorize(w)` as loop metadata. LLVM's O3 pipeline for the host CPU does the
rest — the bias update becomes one `<2 x float>` operation:

```bash
./build/dlc-opt step.dlc --emit-llvm-opt
```
```llvm
define void @k7_mul_sub(ptr nofree readonly captures(none) %0, i64 %1, i64 %2) {
  %4 = load <2 x float>, ptr %"%12", align 4
  %5 = fmul <2 x float> %4, splat (float 1.000000e-01)
  %6 = load <2 x float>, ptr %"%bias", align 4
  %7 = fsub <2 x float> %6, %5
  store <2 x float> %7, ptr %"%bias", align 4
  ret void
}
```

The optimized module is compiled to an object file, which ORC LLJIT links into the process. The object file is
also written to the kernel cache, keyed by a hash of the loop IR and the host CPU: compiling the same step again loads
it and skips LLVM (`minidlc.cache_info()`). The runtime (`lib/runtime/Executable.cpp`) allocates the two regions,
copies the inputs into their slots, launches the kernels in order — splitting each parallel loop across a thread
pool — and returns the loss; the parameters stay in the executable for the next step (`step.sync_params()` copies
them back into the `nn.Parameter`s; `step.save_checkpoint(path)` writes them and the optimizer state to a file any
target can resume from).

## 7. The GPU path

```bash
./build/dlc-opt step.dlc --target=gpu --emit-loopir   # loops bound to blockIdx / threadIdx
./build/dlc-opt step.dlc --target=gpu --emit-ptx      # PTX for sm_75
```

The GPU schedules bind loops to blocks and threads: matmuls stage tiles through shared memory with barriers, row
kernels reduce across a warp with shuffles (and across a block's warps through shared memory, for long rows),
scatters and the loss use atomics. GPU loop passes pad shared tiles against the bank conflicts a warp probe measures,
version partial tiles so that interior blocks run without guards, and check each launch against the hardware's limits
(`--print-gpu-report`). The lockstep interpreter runs the kernels on the host with SIMT semantics
(`target="gpu-sim"`), detecting races that lockstep execution would otherwise hide, and LLVM's NVPTX backend turns
them into PTX. A GPU artifact — the plan as JSON plus the PTX — runs on any NVIDIA GPU through `minidlc/gpu.py` and the
CUDA driver. See `docs/gpu.md` and `docs/passes/gpu_passes.md`.

## 8. How it is checked

- Every graph pass: translation validation on exported models.
- Every VJP: float64 finite differences; whole-model gradients against PyTorch (`test_pytorch_crosscheck.py`).
- Every op through the generated code, on `cpu`, `interp` and `gpu-sim`: differential tests against the reference,
  with odd shapes, broadcasting, empty dimensions, NaN and Inf.
- Training: compiled trajectories against eager training in the reference interpreter and against PyTorch; random
  models from the pipeline fuzzer trained on every target against the reference; checkpoints resumed across targets.
- End to end, nightly: MNIST accuracy on the test set, Char-GPT's held-out loss against the bigram baseline and
  PyTorch.
- The compiler's own invariants: the IR verifier after every pass, the buffer-assignment checker, the bounds and
  dependence checks of every loop kernel, race detection on every simulated GPU launch, the GPU resource check,
  ASan/UBSan in CI.
