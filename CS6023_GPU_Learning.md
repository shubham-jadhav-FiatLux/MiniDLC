# CS6023 — GPU Programming (IIT Madras, Jan–May 2025) — Learning Notes

> Instructor: **Prof. Rupesh Nasre** (IIT Madras, PACE Lab). Lecture 11 is a guest lecture (23–24 Apr 2025).
> This single file holds: lesson-by-lesson notes for all lectures (with the code idioms that matter, my own slide
> annotations folded in, and corrections where a slide is wrong or outdated); a deep review of my four assignments,
> each verified against the provided tests; a full post-mortem of Assignment 4, including a port of the TA's
> evaluator; my TA work for the Aug 2026 offering; interview Q&A; and bridges to AI/LLM infrastructure.
>
> **Evaluation (from Lecture 1):** four assignments (10 + 15 + 15 + 20), MidSem 20, EndSem 20 (open-book, pen and
> paper). Plagiarism lowers the final grade by two letters. "The instructor does not cancel the 8 o'clock class."
> The course is also offered as Nasre's YouTube course (NSM, AICTE).

---

## Table of contents

1. L1 Introduction: why GPUs, HPC platforms
2. L2 Computation: kernels, threads, grids, warps, divergence
3. L3 Memory: the CUDA memory hierarchy, coalescing, shared memory, banks, constant/texture
4. L4 Synchronization: races, locks, atomics, barriers, reduction, scan, concurrent data structures, CPU–GPU sync
5. L5 Functions: qualifiers, host/device code, Thrust
6. L6 Support: debugging and profiling
7. L7 Streams: concurrency, overlap, events, callbacks, cooperative kernels
8. L8 Topics: dynamic parallelism, multi-GPU, peer access, unified memory, PTX, warp-level primitives
9. L9 Case study: parallel graph algorithms on GPUs
10. L11 GPU programming without CUDA: MLIR, libraries, C++ stdpar, Triton, cuTile
11. My assignments (A1–A4): specs, code review, verification, defects, Phase-2 plans
12. My TA work: CS6023 Assignment 1 (Aug 2026), "The Wizard's Lens"
13. Revision and interview questions
14. Bridges to AI infrastructure
15. References

---

## 1. L1 — Introduction: why GPUs

- **The free lunch ended.** Single-thread performance used to rise automatically with clock frequency and
  microarchitecture (pipelining, ILP). Around 2004 the **power wall** hit: power density heading towards
  "nuclear reactor" and "rocket nozzle" levels (Pollack, MICRO-32 1999); Intel cancelled Tejas and Jayhawk because
  of heat. The only option left was **many cores**. In the 40-year trend data, transistors keep doubling every
  ~2 years while frequency and single-thread performance flatten and logical core counts climb.
- **Moore's law:** transistors per chip double about every two years. It's an empirical observation, used by
  the industry to plan, and somewhat self-fulfilling. Jensen Huang declared it dead in 2022.
- HPC is heading for exascale (FLOP/s), and IITM's **AQUA cluster** is one example. Then came deep learning,
  which made GPUs mainstream.
- **Parallel platforms:** shared-memory multicores (2–64 cores; pthreads, OpenMP; NUMA once cores are grouped
  into sockets); distributed clusters (thousands of nodes, no common memory; MPI send/receive/barrier,
  "communication is the key"); **GPUs** (a separate device with a separate address space, thousands of cores,
  massive multithreading, attached via PCIe or NVLink; e.g. GH100 with about 14,500 cores, ~48 TF FP32 and
  ~400 TF TF32 on tensor cores); **FPGAs** (reconfigurable interconnect after manufacturing, good
  processing per watt; Xilinx, Intel, Lattice); **ASICs** (best performance for one application: Ethernet
  switches, video codecs, controllers); and **heterogeneous** systems, where different elements serve
  different needs and data sharing between them is the bottleneck (my note: memory bandwidth).
- **Takeaway:** peak numbers of CPUs and GPUs can't be compared directly. CPUs and GPUs should be used
  *together*, each for the parts of an application it suits.

## 2. L2 — Computation

### 2.1 First kernels and asynchrony

```cuda
__global__ void dkernel() { printf("Hello World.\n"); }   // __global__ = kernel: runs on GPU, called from CPU
int main() { dkernel<<<1, 1>>>(); cudaDeviceSynchronize(); return 0; }   // nvcc hello.cu && ./a.out
```

- Every valid C program is valid CUDA. `nvcc` is the compiler driver and `.cu` is the extension.
- **Kernel launches are asynchronous.** Without `cudaDeviceSynchronize()` the CPU may exit before the GPU
  `printf` buffer is flushed, so nothing prints. Kernels return `void`.
- Several launches queue up in the default stream and run **one after another**. The CPU just enqueues them
  and moves on, and waits at `cudaDeviceSynchronize` (CDS). Only CPU `printf`s and GPU `printf`s can
  interleave. Statements after the CDS always run last.
- `dkernel<<<1, 32>>>()` launches 32 threads: 32 lines with **no guaranteed order**. Each thread writes to its
  own printf buffer, so lines don't get mixed up character by character.
- **Homework seen in the code folder:** find where `nvcc` lives, the CUDA version, and `deviceQuery`
  (`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/extras/demo_suite`).

### 2.2 Host vs device memory

- A discrete GPU has its **own DRAM** across PCIe. A CPU variable can't be dereferenced in a kernel, and a
  `cudaMalloc` pointer can't be dereferenced on the CPU. The programmer keeps two copies and keeps them in sync.
  Using a host global like `const char *msg` inside a kernel fails with "identifier is undefined in device
  code", while `#define msg "..."` works, because a macro is just text.
- **The typical program flow:**
  1. Load data into CPU memory.
  2. `cudaMalloc` and `cudaMemcpy(..., cudaMemcpyHostToDevice)`.
  3. `kernel<<<grid, block>>>(...)`.
  4. `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.
  5. Use the results on the CPU.
- My note: **`cudaMemcpy` goes into the same queue as the kernel**, so it waits for the kernel to finish. No
  explicit synchronization is needed before a blocking `cudaMemcpy` back to the host.
- Classic example: `"Gdkkn\x1fVnqkc-"` with each char `++` on the GPU decodes to `"Hello World."`.

### 2.3 Thread organisation

- A kernel launches a **grid** (a 3D array of **thread blocks**: `gridDim.{x,y,z}`, `blockIdx`). Each block is
  a 3D array of **threads** (`blockDim`, `threadIdx`). `dim3 grid(2,3,4); dim3 block(5,6,7);` launches
  2·3·4 · 5·6·7 = 5040 threads.
- **Global 1D id:** `id = blockIdx.x * blockDim.x + threadIdx.x`.
- **2D id inside a block:** `threadIdx.y * blockDim.x + threadIdx.x` gives row-major ids. The slide's version,
  `threadIdx.x * blockDim.y + threadIdx.y`, also gives unique ids, but in column-major order.
- Typical configurations: 128–1024 threads per block (max **1024**), a few blocks per SM, and millions of
  threads in total.
- **Launching for an arbitrary N:**
  ```cuda
  unsigned nblocks = ceil((float)N / BLOCKSIZE);      // NOT ceil(N / BLOCKSIZE): integer division first!
  kernel<<<nblocks, BLOCKSIZE>>>(vec, N);
  __global__ void kernel(unsigned *v, unsigned n) { unsigned id = blockIdx.x*blockDim.x + threadIdx.x; if (id < n) v[id] = id; }
  ```
  The two bugs this slide asks you to spot are integer division inside `ceil`, and the missing bounds check on
  the last partial block. Idiomatic alternative: `(N + B - 1) / B`. For huge N, use a **grid-stride loop**
  (`for (i = id; i < n; i += gridDim.x*blockDim.x)`).
- **Matrix squaring (N=64)** shows why the mapping matters. CPU: 1.527 ms. v1 (one thread per row,
  `<<<1,N>>>`): 6.391 ms, because there's too little parallelism and each thread does N² work. v2 (one thread
  per element, `<<<N,N>>>`, `ii = id / N`, `jj = id % N`): 0.1 ms. Homework: swapping ii and jj makes
  consecutive threads walk down columns instead of across rows, which breaks coalescing.

### 2.4 The hardware hierarchy: GPU → SM → block → warp → thread

- **SMs** (streaming multiprocessors) run blocks. A block stays on one SM for its whole life.
- **Warp:** 32 consecutive threads of a block executing in SIMD fashion ("SIMT"). Warp threads behave as if
  there were an implicit barrier after every instruction. *It's a misconception that all GPU threads run in
  lock-step; that's only true within a warp.* (Since Volta, threads have independent program counters, so
  even warp lock-step isn't guaranteed; use `__syncwarp()` where it matters.)
- **Thread divergence:** when warp threads take different branches, the hardware runs each path in turn with
  the inactive lanes masked off (NOPs). `if (id % 2) S1; else S2;` runs S1 and S2 one after the other. The
  correctness must not depend on S1 running before S2.
  - Conditions aren't bad in themselves. `if (vectorsize < N)` is uniform. `if (id / 32)` differs between warps
    but is uniform *within* each warp, so it doesn't diverge. **Only conditions that evaluate differently for
    threads of the same warp diverge.**
  - Classic exercise: remove the divergence from `if (x == y) x = z; else x = y;` (given x==y or x==z). Answer:
    `x = y + z - x;` (or `x ^= y ^ z` for integers).
  - `for (ii = 0; ii < id; ++ii)` diverges, because the trip count differs per thread.

## 3. L3 — Memory

### 3.1 The CUDA memory model (typical numbers, from older GPUs; see the table below for modern ones)

| Space | Scope | Lifetime | Latency / bandwidth | Notes |
|---|---|---|---|---|
| Registers | thread | thread | ~1 cycle, ~8 TB/s aggregate | 64K × 32-bit per SM; max 255 per thread; spills go to "local" memory (which lives in DRAM) |
| Shared memory (L1 split) | block | block | ~20–30 cycles, ~1+ TB/s | programmable scratchpad; 48 KB static per block, up to 227 KB on H100 via opt-in |
| L1 cache | SM | — | ~30 cycles | hardware-managed, unified with shared memory |
| L2 cache | all SMs | — | ~200 cycles | 768 KB on old Fermi; 50 MB on H100; fast atomics live here |
| Global (device DRAM/HBM) | grid + host | application | 400–800 cycles | ~200 GB/s on old cards; 3.35 TB/s HBM3 on H100 |
| Constant | grid | application | cached | 64 KB, read-only; **broadcast** when a warp reads one address, serialised otherwise |
| Texture | grid | application | cached | read-only, optimised for 2D spatial locality, with hardware interpolation |

- **Bandwidth vs latency.** Bandwidth is how much data moves per unit time; you raise it with *wide* buses,
  sharing and reuse, compression, and recomputing instead of storing. Latency is how long one access takes; you
  hide it with caches (CPU) or with **massive multithreading** (GPU). While one warp stalls on memory, another
  is scheduled. This can even give superlinear speedups.
- **Locality** matters on GPUs too, but in a GPU-specific spatial form: **coalescing**. For the CPU matrix
  multiply, the i-k-j loop order (unit stride on B and C) took 4.7 s against 9.5 s for i-j-k at 1024³.

### 3.2 Memory coalescing (the most important performance rule)

- If the threads of a warp access words within the same aligned segment (32 words, i.e. 128 B), the requests
  merge into a single transaction (modern GPUs move 32 B **sectors**; one fully coalesced warp access of 4-byte
  words is four sectors). Uncoalesced accesses can need up to 32 transactions per warp.
- My note on the **degree of coalescing (DoC)**: `a[threadIdx.x]` has DoC 32 (one transaction), `a[rand()]` has
  DoC 1, and `a[threadIdx.x * i]` varies with the stride i.
- Patterns: on the CPU, each thread should work through a *contiguous chunk* (`start = id*chunk`); on the GPU,
  consecutive threads should touch consecutive elements (strided by the grid size). Indirect accesses such as
  histograms, graphs and `a[input[id]]` are inherently uncoalesced.
- **AoS vs SoA:** an array of structures gives good CPU locality (all fields of one node together). A
  structure of arrays gives good **GPU coalescing** (neighbouring threads read the same field of neighbouring
  nodes). On the slide, AoS took 61 units and SoA 22.

### 3.3 Shared memory

- `__shared__` declares a per-block, programmer-managed L1 scratchpad. Two blocks never see each other's
  shared memory. My note: index it by `threadIdx`, not the global id.
- **Races in shared memory.** `s = 0` (thread 0), `s += 1` (thread 1), `s += 2` (thread 100), then print: the
  usual output is s=3, but s=1 appears occasionally, because the threads are in different warps and nothing
  orders them. Putting `__syncthreads()` between the steps fixes it. (The barrier between steps 2 and 3 is
  redundant only if those two threads are in the same warp, and since Volta, even that isn't guaranteed.)
- Another bug: writing `str[threadIdx.x]` and having thread 0 print the string needs a `__syncthreads()` before
  the print (redundant only for BLOCKSIZE ≤ 32, and fragile even then).
- **Configuring the L1/shared split:** `cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1 / PreferShared /
  PreferEqual)`. The slide's "On CPU" means the call is made from host code. Modern equivalent:
  `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, pct)`.
- **Dynamic shared memory:** `extern __shared__ int s[];` sized by the third launch parameter,
  `k<<<1, n, n*sizeof(int)>>>()`. For several arrays, carve one buffer up: `int *s1 = s; int *s2 = s + n1;`
  (mind alignment when mixing types). More than 48 KB needs
  `cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`.
- **Bank conflicts:** shared memory has **32 banks** of 4 bytes, and consecutive words go to consecutive banks.
  Accesses from a warp to *different addresses in the same bank* are serialised. Accesses to the *same word*
  are broadcast with no conflict. With 32-bit words and stride s, the conflict degree is gcd(s, 32), so
  `s[32*threadIdx.x]` is a 32-way conflict. Fixes: padding (e.g. `tile[32][33]` for transposes), or swizzled
  layouts (what CUTLASS and FlashAttention use).

### 3.4 Constant and texture memory

- **Constant memory:** `__constant__ unsigned meta[1];` written from the host with
  `cudaMemcpyToSymbol(meta, &hmeta, sizeof(unsigned))`. 64 KB, cached, and ideal when **all threads read the
  same value** (filter weights, parameters). Assignment 2's filters (≤ 4096 values) were made for this.
- **Texture memory:** read-only, cached for 2D locality, with hardware filtering and addressing modes (the SDK
  example rotates an image with `tex2D(tex, tu + 0.5, tv + 0.5)`). The legacy texture-reference API on the
  slides (`texture<...>`, `cudaBindTextureToArray`) has been **removed in CUDA 12**. Use texture **objects**
  (`cudaCreateTextureObject`), or simply `const __restrict__` pointers and `__ldg`.

### 3.5 Compute capability (the slide's table is wrong; use this)

`major.minor` describes the hardware features (`-arch=sm_86`, `__CUDA_ARCH__ == 860` in device code). It's
separate from the CUDA *toolkit* version.

| CC | Architecture | Examples / key additions |
|---|---|---|
| 1.x | Tesla | first CUDA GPUs |
| 2.x | Fermi | caches, ECC |
| 3.x | Kepler | dynamic parallelism (3.5), shuffle |
| 5.x | Maxwell | (my laptop's GPU was Maxwell, `-arch=sm_50`) |
| 6.x | Pascal | unified-memory page faulting, NVLink, HBM2 (P100) |
| 7.0 / 7.2 | Volta | **Tensor Cores**, independent thread scheduling (V100) |
| 7.5 | Turing | INT8/INT4 tensor cores, RT cores |
| 8.0 / 8.6 / 8.7 | Ampere | TF32, BF16, async copy (`cp.async`), sparsity (A100, RTX 30) |
| 8.9 | Ada Lovelace | FP8 (RTX 40, L4, L40) |
| 9.0 | Hopper | TMA, thread-block clusters, wgmma, FP8 Transformer Engine (H100/H200) |
| 10.x | Blackwell (datacenter) | FP4/FP6, tcgen05, TMEM (B200, GB200) |
| 12.x | Blackwell (consumer) | RTX 50 series |

The slide maps Turing to 8, Ampere to 9, Lovelace to 10 and Hopper to 11; those are wrong. Its feature table
is broadly right: dynamic parallelism from 3.5, unified-memory features from 6.x, tensor cores from 7.x,
hardware async copy from 8.0.

Classwork worth redoing: points `struct {int x, y;}`; 4 elements per thread; compute the average x; threads
replace y values above the average; count the replacements with atomics or a reduction.

## 4. L4 — Synchronization

### 4.1 Data races

A **data race** needs (1) multiple threads, (2) a common location, (3) at least one write, and (4) concurrent
execution. Ways to remove one: execute sequentially; privatise or replicate data; separate reads and writes with
a barrier; use mutual exclusion (or atomics).

- `volatile`: the compiler must re-read the variable from memory every time, instead of caching it in a
  register. Flag-based spinning needs this; otherwise `while (!flag);` can become an infinite loop.
- Control and data dependence: `S2; flag = true;` in one thread and `while (!flag); S1;` in the other means
  S2 always happens before S1. Mutual exclusion alone doesn't fix the order.

### 4.2 Mutual exclusion, built step by step (classic two-thread algorithms)

1. **The flag array:** `flag[me] = true; while (flag[other]);`. Gives mutual exclusion (if `volatile`), but
   **deadlocks** if both set their flags at the same time.
2. **The victim variable:** `victim = me; while (victim == me);`. Gives mutual exclusion, but no progress
   unless the other thread keeps taking the lock.
3. **Peterson's lock:** `flag[me] = true; victim = me; while (flag[other] && victim == me);`. Mutual
   exclusion, deadlock-free and starvation-free ("victim = me" is the polite "pehle aap": you first). The order
   of the first two statements matters: swapping them breaks mutual exclusion, as the slide's interleaving
   shows. Peterson's lock doesn't generalise directly to N threads (victim is a single scalar).
4. **Lamport's Bakery algorithm** (N threads, FCFS by ticket): `flag[me] = true; label[me] = 1 + max(label);
   while (∃k≠me: flag[k] && (label[k], k) < (label[me], me));`. The max isn't atomic, so ties are broken by
   thread id. It works without caches, and needs care (fences) on cached, weakly ordered hardware.
5. **On GPUs, locks are usually avoided.** Spinning is very expensive at scale. Locks are possible, but build
   them from atomics.

### 4.3 Atomics

- `++x[0]` is load, increment, store, so with 2 threads the result can be 1. **Atomics** are all-or-none
  read-modify-writes: `atomicAdd, atomicSub, atomicExch, atomicMin, atomicMax, atomicInc, atomicDec,
  atomicCAS, atomicAnd/Or/Xor`. They work on global and shared memory (shared-memory atomics are much
  cheaper), and are `__device__`-scoped by default. There are `_block` and `_system` variants on sm_60+.
- **SSSP relaxation bug:** `if (altdist < dist[n]) dist[n] = altdist;` is a lost-update race. The fix is
  `atomicMin(&dist[n], altdist)`.
- **atomicCAS:** `old = atomicCAS(&var, expected, desired)`. It's the universal primitive: build any RMW with a
  CAS loop (e.g. atomicAdd on double before sm_60, or float atomicMax).
- **Spinlock from CAS:**
  ```cuda
  // Works on CPU / across warps; can HANG for threads of the same warp on pre-Volta GPUs (SIMT reconvergence):
  do { old = atomicCAS(&lock, 0, 1); } while (old != 0);  /* CS */  lock = 0;
  // Warp-safe pattern: do the critical section inside the successful branch of the loop
  do { old = atomicCAS(&lock, 0, 1); if (old == 0) { /* CS */ __threadfence(); atomicExch(&lock, 0); } } while (old != 0);
  ```
  Why the first version hangs: the thread that wins waits at the reconvergence point for its warp-mates, which
  are still spinning for the lock the winner holds. Volta's independent thread scheduling usually avoids this,
  but the warp-safe form is still the one to write.
- **"Single" construct:** `if (atomicCAS(&flag, 0, 1) == 0) { ... }` with no reset, so exactly one thread
  runs the block.
- The tricky `k1` classwork (2 blocks × 32 threads, CAS on `gg`): some thread sets gg to its tid+1, so no
  thread ever sees 0 again and the second printf never runs. The third printf runs for the thread whose
  tid equals the stored value. Takeaway: **atomics from the threads of one warp are serialised too.**
- Worklist push with atomics (`idx = atomicAdd(&wlsize, 1); wl[idx] = item;`), and 1D convolution as warm-up
  classwork.

### 4.4 Barriers and fences

- A barrier is a program point every participant must reach before any continues. The end of a kernel is an
  implicit global barrier. `__syncthreads()` synchronises one block, and **must be reached by all threads of the
  block** (never put it in divergent code or after an early `return`, which is undefined behaviour). For a
  whole grid there's `grid.sync()` from cooperative groups (CUDA 9+, requires `cudaLaunchCooperativeKernel` with
  every block co-resident). Within a warp, use `__syncwarp()`.
- `__syncthreads()` is both a control barrier and a **memory fence** for the block.
  `__threadfence_block()`, `__threadfence()` and `__threadfence_system()` order a thread's writes (and reads) as
  seen at block, device and system scope. A fence **doesn't** force another thread to read the new value: the
  reader needs `volatile` or atomics to avoid stale cached values.
- **Taxonomy:** `__syncthreads` gives control and data sync within a block. Atomics give data sync within a
  block (shared memory) or across all threads (global memory). `__threadfence_block` gives data sync within a
  block, `__threadfence` across all threads. A global barrier gives both kinds of sync for everyone. A
  `while`-loop spin is customisable control sync, useless without data sync. `volatile` gives data visibility
  everywhere.

### 4.5 Reduction (associative operators: sum, min, max, xor, …)

```cuda
for (int off = n/2; off; off /= 2) {             // n power of two, n/2 threads, log2(n) steps
    if (threadIdx.x < off) a[threadIdx.x] += a[threadIdx.x + off];
    __syncthreads();
}
```

- Variants: `a[t] += a[2*off - t - 1]` (folding from the ends), or the "first diagram" tree (pairs of
  neighbours), which has a read/write race unless reads and writes are separated with a temporary and an extra
  barrier.
- Atomics (`atomicAdd(&sum, a[i])`) are simpler but serialise. Tree reduction gives O(log n) depth.
- String concatenation is associative but **not commutative**, so the `a[i] + a[i + n/2]` pairing scrambles the
  order. Reduction schemes that reorder operands need commutativity too.
- **Mark Harris's optimisation ladder** ("Optimizing Parallel Reduction in CUDA"): interleaved addressing with
  divergence → strided index (now with bank conflicts) → **sequential addressing** (above) → first add during
  the global load → unroll the last warp → full unroll with templates → multiple elements per thread. Modern
  code uses **warp shuffles** (`__shfl_down_sync`), then a shared-memory stage across warps, or CUB's
  `BlockReduce`.

### 4.6 Prefix sum (scan)

- Input `4 3 9 3 5 7 3 2`. Inclusive scan gives `4 7 16 19 24 31 34 36`; exclusive scan gives
  `0 4 7 16 19 24 31 34`.
- **Hillis–Steele** (log n steps, O(n log n) work) needs the read/write separation:
  ```cuda
  for (int off = 1; off < n; off *= 2) {
      int tmp; if (tid >= off) tmp = a[tid - off];
      __syncthreads();
      if (tid >= off) a[tid] += tmp;
      __syncthreads();
  }
  ```
  Students' one-barrier variants (Surya's `tid % (2*off)` trick, Prasanna's bit-walking) are on the slides.
- **Blelloch** (work-efficient O(n)) uses an up-sweep (reduce) followed by a down-sweep. The single-pass
  **decoupled look-back** scan (Merrill & Garland, 2016) is what CUB and Thrust use today.
- **Why scan matters:** compaction and worklist insertion without atomics. Each thread knows `nelem[tid]` and
  writes at offset `exclusive_scan[tid]`. Also histogramming, radix sort, area under a curve, Fenwick trees,
  and stream compaction (used in MoE token routing and in top-p sampling).

### 4.7 Global barrier, concurrent structures, CPU–GPU sync

- **Global barrier from atomics:** `__syncthreads()` in each block, then one representative per block does
  `atomicAdd(&count, 1)` and spins until `count == gridDim.x` (the counter must be reset or sense-reversed). It
  **deadlocks** if not every block is resident. That's exactly why `cudaLaunchCooperativeKernel` exists.
- **Concurrent linked-list insertion at the head:**
  - A coarse lock on the list is simple and correct, but sequential.
  - Per-node locks give finer granularity but make correctness harder. `ptr = head; ptr->lock()` is wrong
    because head can change between the read and the lock. Lock `head` first and reload it. A variant inserts
    as the second node, under `head`'s lock.
  - Lock-free insertion uses CAS: `do { old = head; n->next = old; } while (atomicCAS(&head, old, n) != old);`
    (watch for ABA when nodes are reused).
- **CPU–GPU sync:**
  - Ping-ponging a counter with `cudaMemcpy` + kernel + `cudaMemcpy` works, but is slow.
  - **Pinned (page-locked) host memory**, `cudaHostAlloc(&p, size, 0)`, is directly accessible from the
    device (zero-copy through UVA) and is zero-initialised by default.
  - **Persistent kernel:** the GPU spins `while (*counter % 2 == 0);` on host-allocated memory while the CPU
    spins on the other parity. The slide asks whether "10" can fail to print. It can: CPU line 1 increments to
    9, the kernel then runs lines 2–5, and the loop exits before line 6 runs. Correct handshakes need careful
    ordering and memory fences.
- **Hierarchy of barriers:** warp (SIMD) → block (`__syncthreads`) → grid (global barrier) → CPU–GPU
  (`cudaDeviceSynchronize`).
- **Uses of CPU+GPU co-processing:** split matrix stages, graph partitions (each side computes on its part and
  they merge, which is useful when data exceeds GPU memory, e.g. the Twitter graph), and CPU preparing the next
  GPU iteration.

### 4.8 Synchronization patterns (from Nasre's research)

| Situation | Options |
|---|---|
| **Disjoint accesses** (threads push into a worklist) | one atomic per element, O(e); one atomic per thread (reserve a chunk), O(t); prefix sum, O(log t) barriers |
| **Overlapping accesses** (threads try to own sets of elements, e.g. cavities in Delaunay mesh refinement, or unique worklist insertion) | an atomic per element; or **race-and-resolve** (non-atomic mark, barrier, check; prioritised marks) |
| **Benign overlaps** (threads write the same value, e.g. level-synchronous BFS) | atomics aren't needed |

Exploiting **algebraic properties**:
- *Monotonicity:* SSSP distances only decrease, so an atomic-free update with a lost update is later corrected
  by topology-driven re-processing.
- *Idempotency:* duplicate worklist entries are harmless.
- *Associativity:* push information through a prefix sum (scatter–gather).

## 5. L5 — Functions

| Qualifier | Runs on | Callable from |
|---|---|---|
| `__device__` | device | device |
| `__global__` | device | host (and device, with dynamic parallelism); must return `void` |
| `__host__` (default) | host | host |
| `__host__ __device__` | both | both (compiled twice) |

- A `__host__ __device__` function can't use device-only intrinsics like `__syncthreads()`, which isn't
  available on the CPU.
- Host-allocated (pinned) memory "is friends with" `__host__ __device__` functions: both sides can touch it.
  `cudaMalloc` memory can't be accessed on the CPU.
- Variables can't be `__host__`. A file-scope `int counter;` is host-only. Use `__device__ int counter;` for
  device globals; it compiles with a warning if host code touches it.
- Classwork on incrementing an array from host and device: host-centric (a sequential loop inside a `__host__
  __device__` function), device-centric (each thread handles `arr + threadIdx.x`), and Pranav's idea of passing
  `(ptr, count)` so the same function serves both.
- **Thrust** (an STL for GPUs):
  - `host_vector` and `device_vector`; assignment copies with `cudaMemcpy` and resizes automatically;
    `D[i]` on the host is a slow per-element transfer.
  - Algorithms: `fill, sequence, copy, transform` (with `negate`, `modulus`, or functors like `saxpy` that are
    `__host__ __device__`), `replace, find(_if), count(_if), equal, min/max_element, merge, sort, sort_by_key,
    reduce, inclusive/exclusive_scan, set_difference`, and more.
  - Iterator arguments must live on the same device (except `copy`).
  - `thrust::reduce(begin, end, init, binop)` can implement count, given a binop that ignores the left
    operand's meaning.
  - **Scan with an init value:** `exclusive_scan(data, data+10, data, 5, plus)` on
    `{-5,0,2,-3,2,4,0,-1,2,8}` gives `5 0 0 2 -1 1 5 5 4 6`. **`inclusive_scan` with an init value is a
    compile error** in the Thrust versions the course used; newer Thrust/C++ allow it.
    `exclusive_scan(..., 1, maximum)` gives `1 1 1 2 2 4 4 4 4 4`.
  - `set_difference` needs sorted inputs (or a matching comparator such as `greater<int>` for reverse-sorted
    inputs).
  - CUB is the lower-level, block- and warp-level companion to Thrust; both now ship in NVIDIA's CCCL.

## 6. L6 — Support: debugging and profiling

- Debugging parallel code is hard: non-deterministic schedules, varying outputs, huge intermediate states.
- **Always check errors.** Kernel errors are sticky and asynchronous. Pattern:
  `kernel<<<...>>>(); cudaError_t e = cudaGetLastError(); cudaDeviceSynchronize(); e = cudaGetLastError();`
  with `cudaGetErrorName` and `cudaGetErrorString`. Example: dereferencing an unallocated pointer gives `error=77
  cudaErrorIllegalAddress` (named 700 in newer toolkits), and `printf` after it prints nothing. Wrap every call
  in a `CUDA_CHECK(call)` macro.
- **cuda-gdb:** compile with `nvcc -g -G` (`-G` disables device optimisations). Useful commands:
  `info cuda kernels/threads/sms/warps/lanes`, `cuda kernel block thread`, `cuda block 1 thread 0` to switch
  focus, `set cuda break_on_launch application`, and conditional breakpoints (`break f.cu:23 if threadIdx.x == 1
  && i < 5`). Homework: find a use-after-free (`cudaFree(y)` where y aliases x).
- **compute-sanitizer** (replaces cuda-memcheck; not on the slides, but essential): `--tool memcheck |
  racecheck | synccheck | initcheck`. It would have caught the illegal `__syncthreads` in my A2.
- **Profiling:** measure kernel times, memory utilisation, cache misses, divergence and coalescing (about 140
  hardware events, e.g. `gld_inst_8bit`, `inst_executed`). `nvprof` on the slides is deprecated. Today it's
  **Nsight Systems** (`nsys`, timelines) and **Nsight Compute** (`ncu`, per-kernel counters and roofline). Use
  `cudaProfilerStart/Stop` to profile a region.
- **Case study from the slides:** K1 (arithmetic), K2 (`atomicAdd` on a global), K3 (shared-memory sum), each
  launched 100 times. K2 took 39% of the time. Fusing the loops changes little, because the launches still
  dominate. **Kernel fusion** into one K, then launching 500 blocks once instead of 100 launches, drops the GPU
  time from about 485 µs total to 3.7 µs. **Launch overhead and fusion** are lessons that matter directly for
  LLM inference (CUDA Graphs, fused kernels).

## 7. L7 — Streams

- **What can run concurrently:** host computation, device computation, host→device copies, device→host copies,
  intra-device copies, and inter-device copies.
- **Asynchronous with respect to the host:** kernel launches, device-internal copies, small (≤ 64 KB) H2D
  copies, `*Async` copies, and memsets. Set `CUDA_LAUNCH_BLOCKING=1` to serialise everything while debugging.
- Concurrent kernels need compute capability 2.x+ (up to 32–128 resident grids). By default kernels in the
  same (default) stream run in order.
- **Streams** are queues of device work. Operations *within* a stream are ordered; operations *across* streams
  are unordered and may overlap. Launch with the fourth parameter:
  `K<<<g, b, smem, stream>>>`, plus `cudaStreamCreate`.
- **Overlapping copy and compute** with N streams per chunk: `cudaMemcpyAsync(H2D, s[i]); K<<<..., s[i]>>>;
  cudaMemcpyAsync(D2H, s[i]);`. This needs **pinned host memory** and copy engines ("Concurrent copy and kernel
  execution: Yes with 2 copy engines"). The pipeline has fill, full-throttle and drain phases. Loop orderings
  trade latency against buffer space (all H2D, then all K, then all D2H, versus interleaved per chunk).
  Homework: three-stage pipelining for data that doesn't fit on the GPU (D2H of chunk i, K of chunk i+1 and H2D
  of chunk i+2 at once). The same pattern appears in LLM weight and KV offloading.
- **Deadlock questions:** a global barrier across two kernels in different streams can deadlock if they aren't
  co-resident. Two memory transfers can't deadlock each other.
- **Cooperative kernels** (classwork): one kernel processes all N items, unless a second kernel arrives and
  takes half. **Khush's algorithm:** each kernel repeatedly grabs the next chunk of M items with `atomicAdd`
  on a shared counter. This is dynamic work-stealing that generalises to any number of kernels.
- **Stream synchronisation:** `cudaDeviceSynchronize()` waits for all streams; `cudaStreamSynchronize(s)` for
  one stream; `cudaStreamQuery(s)` polls without blocking. **Implicit serialisation points** between streams:
  pinned host allocation, device allocation, device memset, a memcpy within the same device memory, anything
  issued to the **legacy default stream** (it synchronises with all blocking streams; use non-blocking streams
  or `--default-stream per-thread`), and switching the L1/shared configuration. In the slide's quiz, a
  `cudaHostAlloc` between launches serialises them.
- **Priorities:** `cudaStreamCreateWithPriority(&s, flags, prio)`. Lower numbers mean higher priority;
  `cudaDeviceGetStreamPriorityRange` returns the range, e.g. 0 to −5. The slide's analogy: students, staff and
  faculty queues.
- **Callbacks:** `cudaStreamAddCallback(stream, fn, data, 0)` runs `fn` on the host after all earlier work in
  that stream (in the default stream, after all streams). Callbacks block their stream, and **you can't call
  CUDA APIs from inside a callback** (`cudaErrorNotPermitted`). The modern API is `cudaLaunchHostFunc`.
- **Events:**
  - `cudaEventRecord(ev, sA); cudaStreamWaitEvent(sB, ev, 0);` creates a cross-stream dependency (K3 in stream
    B waits for K1 in stream A, while K2 and K3 may overlap).
  - Timing: `cudaEventRecord(start); ...; cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop)`.
  - **Fork-join:** a parent stream records an event; N child streams wait on it and run `work`; each child
    records an event the parent waits on before the postamble.
  - These dependency DAGs are exactly what **CUDA Graphs** capture and replay with near-zero launch overhead.

## 8. L8 — Topics

### 8.1 Dynamic parallelism (DP)

- A kernel launches kernels, which suits nested or recursive parallelism (hierarchical structures, recursion
  with parallel levels, batches of parallel work). Build with `nvcc -arch=sm_35 -rdc=true` (separate
  compilation).
- A parent grid isn't complete until all its children complete. Parent and child share global and constant
  memory; local and shared memory are private, so **never pass pointers to shared or local memory** to a child.
- **Visibility:** parent writes before the launch are visible to the child (use `__syncthreads()` first, so the
  whole block's writes are visible, not just warp 0's). Child writes are visible to the parent only after it
  synchronises.
- **CDP2 change (CUDA 12):** device-side `cudaDeviceSynchronize()` is deprecated from 11.6 and removed for
  sm_90+. Use `cudaStreamTailLaunch` (the child runs after the parent grid completes),
  `cudaStreamFireAndForget`, or `cudaStreamPerThread`.
- Overheads: launch parsing, pending-launch buffers (fixed pool plus a slower variable pool), and parents being
  swapped to memory. Not all nested loops need DP; flattening is often better.

### 8.2 Multi-GPU and peer access

- `cudaGetDeviceCount`, `cudaSetDevice(i)`, and `cudaGetDeviceProperties` (`major.minor`). Allocations, launches,
  streams and events belong to the **current device**. Launching into a stream of a different device fails, but
  a memcpy into another device's stream works. Each device has its own default stream, so kernels on different
  devices run concurrently. Use `Async` copies for overlap.
- Events: `cudaEventRecord` and `cudaEventElapsedTime` need the event (or both events) on the stream's device.
  `cudaStreamWaitEvent` works **across devices**, which gives inter-device dependencies (K4 on GPU1 waits for K1
  on GPU0).
- **Peer access:** `cudaDeviceCanAccessPeer(&ok, d, peer)` and `cudaDeviceEnablePeerAccess(peer, 0)` (directional
  in the API). After that, kernels on one device dereference pointers to another device's memory through UVA
  over NVLink or PCIe. Without it you get the error-77 illegal address. Max 8 peers per device; 64-bit only.
- **Memory kinds:**

| Kind | API | Physically lives | Auto-synced? |
|---|---|---|---|
| Pinned / zero-copy | `cudaHostAlloc` | host | yes (accessed over the bus) |
| UVA | — | one address space for host and all devices | no, but pointers are unambiguous |
| Unified memory | `cudaMallocManaged` | migrates on demand (page faults on Pascal+) | yes |

  Tune unified memory with `cudaMemPrefetchAsync` and `cudaMemAdvise`.

### 8.3 PTX and warp-level primitives

- **PTX** is NVIDIA's virtual ISA (`nvcc --ptx file.cu`); SASS is the real machine code. Special registers:
  `%tid` (threadIdx), `%ntid` (blockDim), `%warpid`, `%nwarpid`, `%smid`, `%nsmid`, `%total_smem_size`.
  Instructions: `bar/barrier` (scoped variants), `membar/fence` (strictness variants), and `atom.{and, or, xor,
  cas, min, …}`. `printf` compiles to a `vprintf` call.
- **Warp primitives** communicate through registers, so they're fast. Each takes a mask, usually
  `0xFFFFFFFF`.
  - `__activemask()` returns the lanes currently executing. It's almost always FFFFFFFF for a full warp, but
    that isn't guaranteed, and inside `if (threadIdx.x == 0)` it returns 1.
  - **Votes:** `__all_sync(mask, p)` and `__any_sync(mask, p)` return 0 or 1; `__ballot_sync(mask, p)` returns
    a 32-bit mask. For `threadIdx.x < 100` in a 128-thread block, ballot gives FFFFFFFF, FFFFFFFF, FFFFFFFF, F
    (the last warp has lanes 96–99). `threadIdx.x % 2 == 0` gives 55555555.
  - `__popc(mask)` counts set bits; `__ffs(mask)` finds the first set bit (1-based from the LSB), which is
    useful for leader election.
- **Warp-aggregated atomics:** replace `if (cond) atomicInc(&ctr, N);` with
  `m = __ballot_sync(FULL, cond); if (lane == 0) atomicAdd(&ctr, __popc(m));`. That's one atomic per warp
  instead of up to 32. To elect the leader among the active lanes instead of lane 0, use `__ffs(m) - 1`.
- Also know (not on the slides): `__shfl_sync, __shfl_down_sync, __shfl_xor_sync` for register-level
  reductions and scans; `__reduce_add_sync` on sm_80+; cooperative groups (`thread_block_tile<32>`,
  `coalesced_threads()`).

## 9. L9 — Case study: parallel graph algorithms on GPUs

- Graphs are everywhere (social networks, roads, molecules, n-body), at enormous scale (Facebook with 2.2B
  users, 100B stars, 100B neurons). Betweenness centrality on a million-node graph takes weeks sequentially.
  Graphs are **irregular**: load imbalance, poor locality.
- Handling large graphs: distributed partitions, compression (keeping retrieval possible), external-memory
  algorithms, parallelism, approximation, and maintaining a graph core. Tools: libraries (Galois, Ligra,
  **LonestarGPU** (Burtscher, Nasre, Pingali), Gunrock) and DSLs (Green-Marl, Elixir, **Falcon** (Nasre's
  group)).
- **Dependence testing:** for `a[2i] = … a[2i+1] …` there's a flow dependence across iterations if the integer
  system `0 ≤ i_w < i_r < 10`, `2i_w = 2i_r + 1` has a solution. It doesn't (the GCD test: 2 doesn't divide 1),
  so the loop is parallel.
- **GPU challenges:** synchronisation (locks are prohibitive, atomics expensive), memory latency (poor locality,
  little caching), thread divergence (degree varies), uncoalesced accesses.
- **Representations:**

| Representation | Space | Good for |
|---|---|---|
| Adjacency matrix | V² | dense graphs; neighbours in O(V) |
| Adjacency list | V + E | sparse graphs |
| Edge list / COO | E | edge-centric algorithms |
| **CSR** (row offsets + concatenated neighbour lists) | V + E | GPU standard; compact to transfer |

- **TAO of parallelism** (Pingali et al., PLDI 2011): an *operator* applied at *active nodes* over a
  *neighbourhood*.
  - **Topology-driven** processing applies the operator to every node each round (Bellman-Ford style). It
    needs no worklists, can use coarse barriers and is easy to write, but does extra work.
  - **Data-driven** processing applies it only to nodes that have work (worklist-based, e.g. delta-stepping).
    It's work-efficient, but needs a central worklist and fine-grained atomics.
- **Data-driven optimisations:** double-buffered worklists (in/out, swapped each round); **hierarchical
  worklists** (block-local in shared memory, flushed to global); **work chunking** (reserve space for many items
  with one atomic); **atomic-free updates via prefix sum**; **work donation** (overloaded threads donate to a
  shared-memory box) for load balance; **variable kernel configuration** that follows the available parallelism
  (the DMR profile on a 50K-triangle mesh).
- **Topology-driven optimisations:** kernel unrolling (process a node and its neighbours in one invocation;
  races must be avoided), exploiting the memory hierarchy (on-chip queue or stack), and an improved memory
  layout (renumber so logically close nodes are physically close).
- **Push vs pull:** push needs atomics (or tolerates lost updates through monotonicity); pull is atomic-free
  but may redo work.
- **Morph algorithms** change the graph itself: Delaunay mesh refinement, points-to analysis, MST,
  survey propagation. They bring sync, memory-allocation and load-imbalance problems (Nasre et al., "Morph
  Algorithms on GPUs", PPoPP 2013).
- **GPU optimisation principles** (Nasre's taxonomy):
  - *Computation:* algorithm selection, work sorting, work chunking, communication onto computation, following
    the parallelism profile, pipelining.
  - *Memory:* kernel transformations, data grouping, exploiting the memory hierarchy.
  - *Synchronization:* avoid it, coarsen it, race-and-resolve, combine.
- **Approximations:** reduced execution (stop after K iterations), partial graph processing, approximate
  attribute values, graph compaction (fewer iterations, fewer edges, merged vertices, fewer distinct values).
- Exercises: dependence tests on `a[2i] = …a[i+1]…` and `a[3+i] = …a[5i]…`; CSR→COO conversion; a degree-count
  kernel; SSSP versus the CUDA SDK.

## 10. L11 — GPU programming without CUDA (guest lecture, Apr 2025)

- CUDA (2006) put power in programmers' hands ("ninja programmers"), but broke abstractions, isn't portable,
  doesn't guarantee performance across versions, and is a maintainer's nightmare (most real code is
  memory-bound and needs **fusion**). Every vendor has its own (HIP, DPC++/SYCL, …).
- **Alternative 1: compilers.** Traditional compiler IRs target scalars and ILP. **MLIR** (Multi-Level IR)
  lowers through *dialects* ("progressive lowering"), optimising at each level and reusing blocks across
  front-ends. MLIR compilers gain a lot from fusion and are more maintainable, but are not yet a full
  replacement for hand-tuned kernels.
- **Alternative 2: libraries.** cuBLAS, cuDNN, cuFFT, cuSPARSE, cuRAND, Thrust/CUB, NCCL, NPP; rocBLAS and
  MIOpen on AMD. Bindings exist for C++ and Python. AI compilers like XLA often **pattern-match to library
  calls** instead of generating code.
- **Alternative 3: better languages.**
  - **C++ `<algorithm>`** (C++11+) grouped into queries, set algorithms, movers, value modifiers, structure
    changes, raw memory, `for_each`/`transform`, the "runes" (`*_n, *_copy, stable_*, is_*, *_if`), and
    permutations. STL algorithms can't change a container's size.
  - **Execution policies (C++17/20):** `seq`, `unseq` (vectorised on the calling thread), `par`, `par_unseq`.
    `std::sort(std::execution::par, …)` with NVIDIA **stdpar** (`nvc++ -stdpar`) runs on the GPU. Rules: no
    `__device__` annotations, the definition and call in the same file, data on the heap (managed memory),
    lambdas instead of function pointers, random-access iterators, and no exceptions on the GPU. Real apps
    (LULESH, STLBM, M-AIA) come close to hand-written CUDA.
  - **Python:** DSLs (TensorFlow, PyTorch, JAX), Numba (JIT), PyCUDA, CuPy, **Triton**, Mojo.
- **Triton** (Tillet, Kung & Cox, MAPL 2019): the kernel is written as a single "program" over **blocks/tiles**,
  and the compiler handles threads, shared memory, coalescing and tensor-core use. It's embedded in Python
  (`@triton.jit`), with pointer arithmetic on blocks (`offsets = pid*BLOCK + tl.arange(0, BLOCK)`),
  `tl.load/tl.store(..., mask=offsets < N)`, launched as `_add[grid](...)` with `grid = (triton.cdiv(N,
  BLOCK),)`. It has NVIDIA and AMD backends via MLIR, and is **the backend of `torch.compile`** (TorchInductor
  generates Triton). Softmax takes about 10 lines and matmul about 25, near cuBLAS for many shapes.
- **NVIDIA cuTile:** block-based programming in Python and C++ (cuTile → Tile IR → GPU). Llama-3.1 has been
  implemented in it. The conclusion: CUDA enabled the AI revolution, and block-level programming (Triton,
  cuTile, and CUTLASS's CuTe DSL) is likely the future of kernel programming.

---

## 11. My assignments

All four assignments live in `Assignments/Assignment_{1..4}`. This machine has no NVIDIA GPU, so I verified
them with (a) saved GPU outputs where they exist, (b) a small **CUDA→CPU emulation shim** (sequential per-thread
execution, valid for kernels without inter-thread communication), and (c) independent CPU reference
implementations. All of this lives in the session scratchpad.

### A1 (10 marks, due 9 Feb 2025) — "Thomas's filters": inverted grayscale + Thomas transformation

- **Task:** given R, G, B matrices (m×n), compute T1 = the grayscale image `⌊(r+g+b)/3⌋` **flipped vertically**
  (`T1(i,j) = gray(m−1−i, j)`), and T2 = `⌊0.5·r⌋ + ⌊√g⌋ + b`. Launch configuration is fixed:
  `ceil(m·n/1024)` blocks of 1024.
- **My code** (`submit/cs24s009.cu`): two 1D kernels. Kernel 1 writes `T1[(rows-1-row)*cols + col] = (r+g+b)/3`;
  kernel 2 writes `floor(0.5*r) + floor(sqrt((float)g)) + b`. Allocations, copies and frees are all correct.
- **Verdict: correct.** Nits: an unnecessary `__shared__ int size` written by every thread (a benign race; a
  register would do), and `sqrt((float)g)` is exact only while g < 2²⁴ (fine for pixel values).

### A2 (15 marks, due 2 Mar 2025) — multi-channel 2D convolution (must use shared memory and coalescing)

- **Task:** an image of H×W×C stored as (H·C)×W; K filters of R×S×C stored as (R·C)×S; zero padding, stride 1,
  odd R and S, centred. Output K stacked H×W maps. This is really **cross-correlation** (no filter flip), as the
  examples show. "**0 marks if sequential or no shared memory.**" Constraints: H, W ≤ 1024; K·R·S·C ≤ 4096.
- **My code** (`cs24s009.cu`, submitted 2 Mar): 32×32 thread blocks, `gridDim.z = K`; each thread loops over
  C·R·S taps.
- **Verification:** with my CPU emulation, it passes all 5 public tests (3×3 to 3×5 images, so tiny).
- **Problems a grader would see:**
  1. **The shared memory is fake.** Each thread writes its own pixel into `sharingIsCaring[ty+1][tx+1]` and
     reads back the same cell. No thread ever reads another thread's data, the halo cells are never filled, and
     every tap costs a global load plus a block barrier. It's slower than a plain global-memory version and
     doesn't meet the "use shared memory" requirement in spirit.
  2. **`__syncthreads()` after an early `return`.** Boundary threads (`row >= h || col >= w`) return, while the
     rest of the block hits barriers R·S·C times. That's undefined behaviour whenever W or H isn't a multiple of
     32 (most real tests).
  3. The filter is read from global memory although it fits in **constant memory** (≤ 4096 × 8 B = 32 KB <
     64 KB), and a warp reads the same filter value, the broadcast case constant memory is built for.
  4. The grid computes `ceil((float)(w+31)/32)` (double rounding), which sometimes launches an extra column of
     blocks. That's harmless but sloppy.
- **Correct design (Phase 2):** filters in `__constant__`; each block cooperatively loads a
  `(32+R−1)×(32+S−1)` halo tile of one channel into shared memory (coalesced row loads), `__syncthreads()`,
  then each thread accumulates its R×S taps from shared memory. Loop over channels, double-buffering the tile.
  No early returns before barriers: guard the stores instead.

### A3 (15 marks, due ~6 Apr 2025) — minimum-cost road network (MST with terrain multipliers)

- **Task:** edges `u v wt type` with multipliers green ×2, traffic ×5, dept ×3, normal ×1. Output the MST weight
  mod 1e9+7. Allowed algorithms are parallel Prim, Borůvka, Kruskal and the shortest-path family. Execution time
  is evaluated, and sequential code is "harshly penalised".
- **My code** (`original_submission/cs24s009.cu`): **Borůvka** on the GPU. Each round:
  1. reset the per-component best edge;
  2. one thread per edge finds both roots (`sbj_find` with path compression) and does
     `atomicMin(&best[root], (weight<<32) | edgeIdx)`, packing weight and edge index into 64 bits, which gives
     a consistent tie-break, so no cycles;
  3. one thread per vertex merges via `atomicCAS(&parent[max(u,v)], max, min)`, adds the weight and counts
     changes;

  and it repeats until no merge happens.
- **Verification:**
  - I wrote a 64-bit Kruskal reference. It reproduces all **30/30** expected outputs.
  - My saved GPU outputs (`cs24s009/output*.txt`, from the timing-printing variant) match all 30 answers,
    with runtimes from 2 ms to 0.53 s (test 28: 753K vertices, 2.9M edges).
  - **Verdict: correct and fast.**
- **Latent issues:**
  - `mst_cost` is a 32-bit `int` with `atomicAdd`, and `wt * multiplier` is computed in `int`. The largest
    public MST sum is 826,619,038, under 2³¹, but the "mod 1e9+7" in the spec signals larger hidden sums. Use
    `unsigned long long` and reduce modulo at the end.
  - Input parsing uses `cin.peek()` to make the type optional. A trailing space on a type-less line would make
    it swallow the next line's first token.
  - Union-find with concurrent path compression is a benign race here (pointers only ever point to smaller ids),
    but it deserves a comment explaining why it's safe.

### A4 (20 marks, due 5 May 2025) — "Attack on Titan: Evacuation" ⚠️ scored about 0

**Problem.** An undirected graph of cities (up to 100K) and roads (up to 1M) with length (km) and capacity
(people per batch). Shelters have capacities. Populated cities (up to 10K) have prime-age and elderly
populations. Elderly can walk at most `max_distance_elderly` km. Everyone walks at 5 km/h, i.e. **12 minutes per
km**. A road of length L used by P people takes `12·L·⌈P/cap⌉` minutes (people move in batches), and one city's
group uses a road exclusively. Contention priority: earlier arrival, then the larger *moving* population, then
the lower city id. Output, for each populated city, **one walk** (a path) and **drops** (city, prime, elderly)
in path order. People dropped at a shelter count as saved, up to its capacity; overflow by x loses x more
(capped at the capacity). Score per test = `1.0·(P − Pmin + α)/(Pmax − Pmin + α) + 0.4·(Tmax − T + β)/(Tmax −
Tmin + β)`, normalised by the best submission. That's **people first, time second**, over 40 hidden tests
(15 small, 15 medium, 10 stress). The algorithm **must run on the GPU** (Q&A #20), and there's a 10-minute
limit.

**What the evaluator actually does.** I read `A4_Test/submissions/evaluation.cu` and ported it faithfully to
C++ (scratchpad `a4tools/eval_port.cpp`):
- **Output format it parses:** the first `num_populated` lines each hold one path (space-separated city ids);
  the next `num_populated` lines each hold *all* drops of that path as `city prime elderly` triples **on one
  line**.
- It simulates in lockstep ticks of 12 minutes, with three `grid.sync()` phases per tick. Final time =
  12 × number of ticks until every populated city has dropped everyone.
- **Quirks a good solution must respect:**
  1. Batches are computed from **prime-age only** (`batchPeopleLeft = evacRem`); elderly don't use capacity.
  2. `distanceTravelled` increments **every km of every batch crossing**, so with ⌈prime/cap⌉ batches the
     elderly "distance" grows by L·⌈prime/cap⌉ per road. Elderly dropped after their budget is exceeded aren't
     counted. If elderly are still undroppable when the path ends, the city never finishes, so **the evaluator
     never terminates → timeout → 0 for the whole test.**
  3. Only one drop can happen per node arrival, in list order. Drops at `path[0]` are applied at t=0.
  4. Edge lookup takes the **first road in input order** between u and v. With parallel edges, the first one's
     length and capacity apply, even if you planned on a better parallel edge.
  5. Overfilled shelter: `saved = cap − excess`, **not clamped**, so a shelter filled to more than 2× its
     capacity contributes *negative* people.
  6. The contention key is `((arrival·10000) + (10000 − primeRemaining))·10000 + cityId`, and the release at
     time T lets the next group start at T.
  7. Any invalid path (missing edge) or leftover population makes the result `-1 -1`, or a hang.

**What I submitted** (`Assignment_4/main.cu`, identical to `A4_Test/submissions/CS24S009/main.cu`): for each
populated city, a CPU Dijkstra to the nearest shelter for elderly (bounded by max distance) and another for
prime-age, then a trivial GPU "merge" kernel.

**Verification:** I ran a CPU port of my algorithm through the ported evaluator on all 40 test inputs.

| Variant | Result on the 40 tests |
|---|---|
| **As submitted (my output format)** | the evaluator **hangs on 40/40**, so a timeout and score 0 everywhere. This matches my `scores/CS24S009.txt` (empty or MISSING_OUTPUT). |
| Same algorithm, correct format | 7/40 still hang; 21/40 give **negative** saved (everyone overflows the nearest shelter); 12/40 positive |
| "Everyone stays home" baseline | valid on 40/40 (time 0), and better than my algorithm on most tests |

**Defects** (line numbers refer to `main.cu`):
1. **Wrong output format** (l.238–254). It prints `path_len`, the path, `num_drops`, then one drop per line.
   The Q&A explicitly said not to break the provided output code. This alone zeroes every test.
2. **Broken path merge** (l.99–102). It appends `pp[1:]` (the prime path's suffix from the source's neighbour)
   after the end of the elderly path, which produces non-adjacent hops. When the elderly path is just `[src]`,
   the prime path isn't appended at all, but prime is still "dropped" at a shelter that isn't on the path, so
   that city never finishes.
3. **No capacity awareness:** every city floods its nearest shelter, and with the unclamped penalty, saved
   goes deeply negative.
4. No modelling of contention or time; elderly budget computed on graph distance, not the evaluator's
   batch-inflated distance.
5. **The core algorithm runs on the CPU** (Dijkstra with OpenMP). The rules require GPU computation, so a
   sequential-code penalty applies. The GPU kernel only copies arrays.
6. `printf("/\n")` **inside the Dijkstra loop** (l.52): about 2×10⁸ lines on stress tests.
7. `d_paths(num_pop * num_cities)` allocates up to 10⁹ ints (4 GB), and `auto h_paths = d_paths` makes **another
   device vector** (not a host copy), so the output loop does one PCIe transfer per element.
8. A latent OpenMP bug: `ptr_eld[i+1] = ptr_eld[i] + …` inside `omp critical` assumes in-order iterations. It's
   harmless only because the grader's `nvcc` line doesn't enable OpenMP.

**Phase-2 plan for A4** (a strong, resume-worthy redo):
- **Infrastructure:** the evaluator port (done), a CUDA→CPU shim for logic testing, and a small local benchmark
  over all 40 inputs. Final GPU runs need an NVIDIA machine (Colab, Kaggle or AQUA).
- **Algorithm:**
  1. A GPU multi-source shortest-path pass (delta-stepping or Bellman-Ford with `atomicMin`, using the
     first-edge semantics) from shelters, plus per-city k-nearest-shelter lists.
  2. **Capacity-aware assignment** of people to shelters (greedy by distance or regret, using atomic
     reservations on remaining capacity). Ideally solve it as a transportation or min-cost-flow problem, since
     total shelter capacity ≈ total population in these tests.
  3. Elderly drops planned against the evaluator's batch-inflated distance (drop elderly first, at the first
     reachable shelter).
  4. **One walk per city that can visit several shelters**, dropping a share at each.
  5. Never overflow.
  6. A contention-aware time estimate from simulating the plan on the GPU, followed by local search (reroute
     the critical-path cities).
- **Deliverable:** correct format, 40/40 valid, near-maximal saved people, then minimise time.

---

## 12. My TA work — CS6023 Assignment 1 (Aug 2026): "The Wizard's Lens"

In `TA_Work/` I designed and built Assignment 1 for the next offering (released 1 Aug 2026, due 23 Aug 2026,
10 marks): an **ImageNet-style preprocessing pipeline** in four kernels.
1. **Grayscale**, BT.601 luma: `0.299R + 0.587G + 0.114B`, kept as float.
2. **Bilinear resize** with align-corners (`s = (H−1)/(Hr−1)`; the four-tap blend with clamped `x1, y1`).
3. **Centre crop.**
4. **Normalise** (mean/std).

This mirrors `torchvision.transforms`. The instructor toolkit includes the LaTeX spec, released boilerplate
(`pipeline.cu`), a commented `solution.cu`, a tolerance-based comparator, public and hidden tests, a NumPy
reference generator, **`cu2cpp.py` plus `cuda_cpu_shim.h` to grade without a GPU**, 10 deliberately broken
submissions to self-test the grader, and a batch grader (`evaluate.sh/.py`). The grader writes a CSV with
integrity checks (protected regions modified?) and GPU-usage checks, both reported but not auto-penalised.
**Resume line:** *"Teaching Assistant, CS6023 GPU Programming (IIT Madras): designed an image-preprocessing
CUDA assignment and built an automated grading pipeline with integrity and GPU-usage checks and CPU-emulated
grading."*

---

## 13. Revision and interview questions (with short answers)

1. **Why is `ceil(N / BLOCKSIZE)` wrong?** Integer division truncates before `ceil`. Use `(N + B − 1)/B`, and
   bounds-check inside the kernel.
2. **When does an `if` cause divergence?** Only when lanes of the *same warp* evaluate it differently.
3. **What is coalescing, and how do you get it?** Neighbouring lanes access neighbouring addresses in one aligned
   segment. Use SoA layouts, `id`-contiguous indexing, and shared memory to transpose irregular patterns.
4. **What are bank conflicts, and the classic fix?** Several lanes hit different words in the same bank. Pad to
   `[32][33]` or swizzle.
5. **Why does a CAS spinlock hang inside a warp on pre-Volta GPUs?** SIMT reconvergence: the lock holder waits
   for warp-mates spinning on the lock it holds. Put the critical section inside the success branch.
6. **`__syncthreads` vs `__threadfence`?** A block barrier (control plus memory ordering for the block) versus a
   pure memory-ordering fence at device scope (no waiting for other threads).
7. **How do you build a grid-wide barrier safely?** Cooperative launch plus `grid.sync()`. Hand-rolled atomic
   barriers deadlock when not all blocks are resident.
8. **Reduction complexity on GPUs?** O(n) work, O(log n) depth. Use sequential addressing, first-add-on-load,
   warp-shuffle the last warp, and handle several elements per thread.
9. **Inclusive vs exclusive scan, and one application?** Exclusive scan gives write offsets for stream
   compaction or worklist insertion without atomics.
10. **Warp-aggregated atomics?** Ballot, popcount, and a leader does one `atomicAdd` for the whole warp.
11. **What's needed for H2D copies to overlap kernels?** Pinned host memory, `cudaMemcpyAsync`, non-default
    streams, and a GPU with copy engines.
12. **What implicitly serialises streams?** The legacy default stream, `cudaMalloc`/`cudaHostAlloc`, memsets,
    and L1/shared configuration changes.
13. **Events vs callbacks?** Events give cross-stream dependencies and timing. Callbacks run host code after
    stream work, and can't call CUDA APIs.
14. **Dynamic parallelism memory rules?** Parent writes before a launch are visible to the child; child writes
    are visible to the parent after it synchronises; never pass shared or local pointers.
15. **Topology- vs data-driven graph processing?** All nodes every round (simple, wasteful) versus a worklist of
    active nodes (efficient, needs atomics or scans).
16. **How would you make my A2 genuinely use shared memory?** A cooperative halo-tile load, barrier, compute
    from the tile, with filters in constant memory.
17. **Why is the Borůvka packed `(weight<<32)|idx` key important?** It gives a strict total order on edges, so
    simultaneous minimum-edge selection can't form cycles.
18. **Roofline for a kernel?** `min(peak FLOPs, AI × BW)`. Compute the arithmetic intensity
    (FLOPs/byte) and see whether you're memory- or compute-bound.

---

## 14. Bridges to AI infrastructure (this is the most directly relevant course)

| CS6023 concept | LLM/AI-systems counterpart |
|---|---|
| Shared-memory tiling, coalescing, bank-conflict-free layouts | **GEMM** (CUTLASS/CuTe, cuBLAS), and **FlashAttention** (tile Q, K, V into SRAM, *online softmax*, never materialise the N×N matrix). FA-2 adds work partitioning; FA-3 on Hopper uses TMA, wgmma and warp specialisation |
| Reductions and warp shuffles | softmax, LayerNorm/RMSNorm, and logsumexp kernels; fused residual + norm kernels |
| Prefix sum and stream compaction | **MoE token routing** (histogram the tokens per expert, scan, then scatter or permute); top-k/top-p sampling; ragged batching offsets (`cu_seqlens` in FlashAttention's varlen API) |
| Atomics and contention (warp aggregation) | KV-cache block allocators, page-table updates (PagedAttention), split-K GEMM accumulation |
| Streams, events, overlap | overlapping communication with compute in data, tensor and pipeline parallelism; prefetching or offloading weights and KV (FlexGen, DeepSpeed ZeRO-Infinity); **CUDA Graphs** to kill launch overhead in decode loops (vLLM and TensorRT-LLM capture graphs per batch size) |
| Kernel fusion lesson (L6) | `torch.compile` / Inductor, TensorRT, and XLA fusion; fused MLP and attention kernels; why decode is launch- and bandwidth-bound |
| Multi-GPU, peer access, UVA | **NCCL** collectives (ring and tree all-reduce over NVLink/NVSwitch), custom all-reduce kernels using P2P (vLLM, TensorRT-LLM), and expert parallelism all-to-all |
| Unified memory and pinned memory | CPU offload of optimizer states and KV; Grace-Hopper coherent memory |
| Dynamic parallelism and persistent kernels (Khush's scheme) | persistent "megakernels" for LLM decode (e.g. Mirage, ThunderKittens-style), stream-K GEMM load balancing |
| Triton and MLIR (L11) | writing custom fused kernels in Triton for research (e.g. Liger kernels, FlashAttention in Triton); TorchInductor codegen |
| Graph algorithms and irregularity (L9) | GNN training (sparse aggregation, CSR SpMM), sparse attention, dynamic batching as irregular parallelism |
| The A4 evacuation problem (paths, capacities, contention, simulation) | **Request routing and scheduling in LLM serving clusters**: capacity-constrained assignment of requests to replicas (KV-memory capacity is a "shelter capacity"), contention on shared links (all-to-all in MoE), and simulation-in-the-loop optimisation (like Vidur) |

**Project seeds from this course:**
1. **A from-scratch FlashAttention-style fused attention kernel** (CUDA and Triton versions) with a roofline
   analysis and a comparison to PyTorch SDPA. This is the canonical "systems person moving into AI" project.
2. **A paged KV-cache allocator plus attention kernel** (a mini vLLM PagedAttention) with a tiny decoding
   engine using CUDA Graphs. Measure tokens/s against batch size.
3. **The A4 redo as a showcase:** a GPU-accelerated, capacity-constrained evacuation and routing planner
   validated by a GPU simulator. Frame it as a general "flows over networks with contention" optimiser, which
   maps onto request routing and MoE all-to-all scheduling.

---

## 15. References

- R. Nasre, *GPU Programming* lecture slides and YouTube course (IIT Madras). D. Kirk & W. Hwu, *Programming
  Massively Parallel Processors*, 4th ed. (2022). NVIDIA, *CUDA C++ Programming Guide* and *Best Practices
  Guide*.
- M. Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA). M. Harris, S. Sengupta, J. Owens, "Parallel Prefix
  Sum (Scan) with CUDA", *GPU Gems 3* Ch. 39. D. Merrill, M. Garland, "Single-pass Parallel Prefix Scan with
  Decoupled Look-back" (2016).
- G. Peterson, "Myths About the Mutual Exclusion Problem" (1981). L. Lamport, "A New Solution of Dijkstra's
  Concurrent Programming Problem" (Bakery), CACM 1974.
- K. Pingali et al., "The TAO of Parallelism in Algorithms", PLDI 2011. R. Nasre, M. Burtscher, K. Pingali,
  "Morph Algorithms on GPUs", PPoPP 2013; "Data-driven versus Topology-driven Irregular Computations on GPUs",
  IPDPS 2013; "Atomic-free Irregular Computations on GPUs", GPGPU 2013. M. Burtscher, R. Nasre, K. Pingali,
  "A Quantitative Study of Irregular Programs on GPUs", IISWC 2012 (LonestarGPU). Y. Wang et al., "Gunrock",
  PPoPP 2016.
- O. Borůvka (1926); J. Kruskal (1956); delta-stepping: U. Meyer, P. Sanders, J. Algorithms 2003.
- P. Tillet, H. T. Kung, D. Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network
  Computations", MAPL 2019. C. Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific
  Computation", CGO 2021. NVIDIA stdpar blog (Olsen, Lopez, Lelbach).
- T. Dao et al., "FlashAttention" (NeurIPS 2022), "FlashAttention-2" (ICLR 2024), "FlashAttention-3" (2024).
  W. Kwon et al., "PagedAttention / vLLM", SOSP 2023.
