# ID5130 — Parallel Scientific Computing (IIT Madras, Jan–May 2026) — Learning Notes

> Instructor: **Prof. Kameswararao Anupindi**, Department of Mechanical Engineering, IIT Madras.
> Textbooks behind the slides: P. Pacheco, *An Introduction to Parallel Programming* (2011) for OpenMP, MPI and
> examples; P. Moin, *Fundamentals of Engineering Numerical Analysis* for the numerics; plus the OpenACC material
> (Juckeland; Hwu et al.).
>
> This file holds: the intro; numerical methods (from my own handwritten class notes,
> `LectureSlides/Numerical_Methods.pdf`); OpenMP, MPI and OpenACC lesson by lesson; the ten tutorials; a review of my three
> assignments, which I **re-ran** where possible (this machine has `gcc -fopenmp` and `mpicc`); a deep review of my
> **final project**, a hybrid MPI+OpenMP ray tracer on the AQUA cluster; revision questions; and bridges to AI
> infrastructure.
>
> **Project rubric** (15 marks): work + code 5, analysis of results (problem-specific plots *and* parallel metrics) 5,
> report writing (ASME conf template; problem statement, objectives, abstract, nomenclature, conclusions, references,
> equations, flowchart, algorithms) 5. Each assignment is out of 100.

---

## Table of contents

1. Introduction: why parallel computing, how to write parallel programs, classification
2. Numerical methods: finite differences, compact schemes, tridiagonal solvers (TDMA, recursive doubling,
   cyclic reduction), iterative solvers (Jacobi, Gauss–Seidel, SOR)
3. OpenMP
4. MPI
5. OpenACC
6. Performance analysis: speedup, efficiency, Amdahl, Gustafson, Karp–Flatt
7. Tutorials 1–10
8. My assignments (A1–A3): what I did, verification, issues, fixes
9. My final project: parallel ray tracer for analytical objects with CSG (hybrid MPI+OpenMP on AQUA)
10. Revision questions
11. Bridges to AI infrastructure
12. References

---

## 1. Introduction

- **Why parallel?** Single-µP performance grew about 50%/year from 1986 to 2002 (≈57.7× per decade). After 2002 it
  slowed to about 20%/year (≈6.2× per decade). Around 2005 manufacturers put **multiple processors on a single chip**
  instead of building faster monolithic ones. Making transistors smaller and faster raises power density, and an IC
  that gets too hot is unreliable; air cooling hits its limit.
- **Why ever more compute?** Climate and weather modelling, energy (wind turbines, batteries, solar cells), data
  analysis (search, genomics, LHC, medical imaging, astronomy), protein folding (misfolding linked to Parkinson's and
  Alzheimer's), drug discovery, CFD and turbulence ("unsolved problem in physics"), cell modelling, materials and
  superconductivity.
- **Why can't we just write translation programs** that convert serial code into parallel code? Take the global
  sum. The obvious parallel version (each core sums its share, then every core sends to core 0) is a straightforward
  generalisation. The efficient **tree-structured** sum bears little resemblance to the serial loop. A translator
  is unlikely to discover it, and the harder the program, the less likely. **We must write explicitly parallel
  programs.**
- **Task vs data parallelism** (5 TAs grading 100 scripts of 5 questions): each TA grades one question (task
  parallelism), or each TA grades 20 complete scripts (data parallelism).
- **Coordination** between cores happens through *communication*, *load balancing* and *synchronization*.
- **Classification:** Flynn's taxonomy (SISD, SIMD, MISD, MIMD), based on the multiplicity of instruction and data
  streams. **Shared-memory** systems (UMA, or NUMA once there are sockets) are programmed with OpenMP and pthreads.
  **Distributed-memory** systems (clusters) are programmed with MPI. **Accelerators** (GPUs) are programmed with
  OpenACC, CUDA or OpenCL.

## 2. Numerical methods (from my handwritten class notes)

### 2.1 Matrix kernels with OpenMP

- **Addition** `C = A + B`: no loop-carried or data dependence.
  `#pragma omp parallel for default(none) shared(A,B,C) private(i,j)` parallelises only the outer loop. To
  parallelise both loops, use `collapse(2)` (or two separate constructs).
- **Multiplication** `C[i][j] += A[i][k]*B[k][j]`: the `i` and `j` loops are independent; `k` is a reduction into
  `C[i][j]`. A parallel region with `#pragma omp for` on the outer loop works, and so does putting it on the
  `j` loop. The innermost `k` loop can only be parallelised as a reduction. Initialise C to zero first.

### 2.2 Explicit finite differences (Taylor series)

On a uniform grid with spacing `h`, expand the Taylor series:
`f_{i+1} = f_i + h f'_i + h²/2! f''_i + h³/3! f'''_i + …` and
`f_{i-1} = f_i − h f'_i + h²/2! f''_i − h³/3! f'''_i + …`

| Scheme | Formula | Order of truncation error |
|---|---|---|
| Forward (FDS) | `f'_i ≈ (f_{i+1} − f_i)/h` | O(h) |
| Backward (BDS) | `f'_i ≈ (f_i − f_{i−1})/h` | O(h) |
| Central (CDS) | `f'_i ≈ (f_{i+1} − f_{i−1})/(2h)` | O(h²): halving h quarters the error |
| 4th-order central | `f'_i ≈ (f_{i−2} − 8f_{i−1} + 8f_{i+1} − f_{i+2})/(12h)` | O(h⁴): halving h cuts the error 16× |
| Second derivative | `f''_i ≈ (f_{i+1} − 2f_i + f_{i−1})/h²` | O(h²) |

- Higher-order stencils need points outside the domain near the boundary, so use **one-sided formulas** there (e.g.
  first-order FDS at i=0 and BDS at i=n).
- In parallel code the loop over i is fully parallel (`#pragma omp parallel for … private(i)`), with the boundary
  `if` branches handled inside. It extends directly to 2D and 3D.

### 2.3 Implicit / compact (Padé) schemes

- Instead of widening the stencil, couple **derivative values** at neighbouring points:
  `f'_{i−1} + 4f'_i + f'_{i+1} = (3/h)(f_{i+1} − f_{i−1}) + O(h⁴)`, a 4th-order scheme on a 3-point stencil.
  Boundary closures: `f'_0 + 2f'_1 = (1/h)(−(5/2)f_0 + 2f_1 + (1/2)f_2)` and
  `f'_n + 2f'_{n−1} = (1/h)((5/2)f_n − 2f_{n−1} − (1/2)f_{n−2})`.
- The result is higher accuracy and better **spectral resolution** (it captures oscillations at lower grid
  resolution), which is why DNS and LES use them. The price is solving a **tridiagonal linear system** for all f'
  at once.

### 2.4 Tridiagonal (banded) systems

- Banded matrix: `a_ij = 0` for `|i−j| > r`. With r = 1 the matrix is **tridiagonal**: sub-diagonal `a`, diagonal
  `b`, super-diagonal `c`.
- **Thomas algorithm (TDMA)** = Gaussian elimination / LU specialised to tridiagonal matrices, O(n):
  - LU: `u_1 = b_1`; `l_{k+1} = a_{k+1}/u_k`; `u_{k+1} = b_{k+1} − l_{k+1} c_k`.
  - Forward substitution `Lz = y`, then back substitution `Ux = z`.
  - It's inherently sequential (loop-carried dependence), so it's great for serial code or for many independent
    systems in parallel (e.g. one per grid line in 2D/3D), but a single system can't be parallelised this way.
- **Parallel tridiagonal solvers** (Hockney & Golub, ~1965). Both are elimination-based and require an SPD or
  **diagonally dominant** matrix (`|b_i| > |a_i| + |c_i|`).
  - **Recursive doubling** (as taught, Stone's variant): at step k, *every* equation i is combined with equations
    `i ± 2^{k−1}` to eliminate `x_{i±2^{k−1}}`. The sub- and super-diagonals move away from the main diagonal
    (stride doubles), and after `⌈log₂ n⌉` steps the matrix is **diagonal**: `x_i = y_i^{(N)} / b_i^{(N)}`. There is
    no back-substitution. Updates with `α_i = −a_i^{(k−1)}/b_{i−2^{k−1}}`, `β_i = −c_i^{(k−1)}/b_{i+2^{k−1}}`:
    ```
    a_i^(k) = α_i a_{i−2^{k−1}}         c_i^(k) = β_i c_{i+2^{k−1}}
    b_i^(k) = b_i + α_i c_{i−2^{k−1}} + β_i a_{i+2^{k−1}}
    y_i^(k) = y_i + α_i y_{i−2^{k−1}} + β_i y_{i+2^{k−1}}
    ```
    Out-of-range terms are zero. Work is **O(n log n)** (more than TDMA), but each step is fully parallel over i.
  - **Cyclic reduction:** at each level only *half* the equations (say the even ones) are updated, so the system
    halves each level. Solve the final small system, then **back-substitute** level by level. Total work O(n), depth
    O(log n).
- **Worked recursive-doubling example** (from my A3 hand calculation, done the way the course wants):
  `3x₁−x₂=2; −x₁+3x₂−x₃=1; −x₂+3x₃−x₄=1; −x₃+3x₄=2`.
  - Step 1 (stride 1), all i: α=1/3 and β=1/3 wherever a neighbour exists. Result: `b = (8/3, 7/3, 7/3, 8/3)`,
    `y = (7/3, 2, 2, 7/3)`. Equation i now couples only to `x_{i±2}`: `a₃=a₄=−1/3`, `c₁=c₂=−1/3`.
  - Step 2 (stride 2): α₃=α₄ from equations 1–2 and β₁=β₂ from equations 3–4 (1/7 and 1/8). Result:
    `b = y = (55/21, 55/24, 55/24, 55/21)`. **The system is diagonal**, so `x = (1,1,1,1)`.

### 2.5 Iterative methods for `Ax = b`

- Direct methods (Gaussian elimination, LU, TDMA variants) store and factor the whole matrix, which is expensive for
  large sparse systems. **Iterative methods** need less storage, only matrix-vector products over the non-zeros,
  and can be faster. They are compared by the iterations needed to converge.
- **General splitting** `A = M − N`, where M is easy to invert:
  `x^{(k+1)} = M⁻¹N x^{(k)} + M⁻¹b = C x^{(k)} + d`. The scheme converges if the **spectral radius** `ρ(C) < 1`.
- **Jacobi:** `A = D − L − R`, `M = D`.
  `x_i^{(k+1)} = (b_i − Σ_{j≠i} a_ij x_j^{(k)}) / a_ii`. Every i is independent, so it's **trivially parallel**.
  Use a separate `xnew` and copy back (or swap buffers). Stop on the residual `r = b − Ax` instead of a fixed
  iteration count.
- **Gauss–Seidel:** uses the newest values, `M = D − L`, `C_GS = (D−L)⁻¹R`:
  `x_i^{(k+1)} = (b_i − Σ_{j<i} a_ij x_j^{(k+1)} − Σ_{j>i} a_ij x_j^{(k)}) / a_ii`. It converges about twice as fast
  as Jacobi on model problems, but it's sequential in i. **Red-black ordering** (see CS6400 L3) restores parallelism.
- **SOR:** `x_i ← (1−ω)x_i + ω·GS_update`, with `ω ∈ (1,2)` (over-relaxation).
- Jacobi and GS are too slow on their own for practical large problems, but they are used as **smoothers and
  preconditioners** inside conjugate gradient and **multigrid**.
- **Convergence facts to remember** (the 5-point Laplacian on an N×N grid, Δ = 1/N): `ρ_Jacobi = cos(πΔ) ≈ 1 −
  π²Δ²/2`. So the iterations needed scale as **O(1/Δ²) = O(N²)**. `ρ_GS = ρ_J²`; optimal SOR is about
  `1 − 2πΔ`, giving O(N) iterations. **Stopping on a small successive difference `max|x^{k+1} − x^k| < tol` is
  dangerous when ρ ≈ 1**: the change per iteration is about `(1−ρ)·error`, so the true error ≈ `tol/(1−ρ)`. This
  is exactly what went wrong in my A2 (Section 8.2).

## 3. OpenMP (shared memory)

- **Model:** a directive-based API (C/C++ `#pragma omp`, Fortran `!$OMP`) for incremental parallelisation of serial
  code. Compile with `gcc -fopenmp` (`icc -qopenmp`, `icx -fiopenmp`, `gfortran -fopenmp` with `-cpp`). Guard
  OpenMP-specific code with `#ifdef _OPENMP` so the same source builds serially.
- **Fork-join:** the master thread forks a team at `#pragma omp parallel`. Every thread runs the block, there's an
  **implicit barrier** at the end, then they join. Output order is non-deterministic.
- **Thread count:** the `num_threads(n)` clause, `omp_set_num_threads()`, or `OMP_NUM_THREADS`. Query with
  `omp_get_thread_num()` and `omp_get_num_threads()`; `omp_in_parallel()` also exists.
- **Trapezoidal rule** (Pacheco's running example): `T = h[f(a)/2 + Σ_{i=1}^{n−1} f(x_i) + f(b)/2]`, `h = (b−a)/n`.
  Each thread integrates `[local_a, local_b]` with `local_n = n/p`, then the results are combined. A shared
  `global_result += my_result` is a **race**:
  - `#pragma omp critical` fixes it, but serialises;
  - the **reduction** clause is cleaner and faster: `reduction(+: total)`. The reduction variable is private
    inside the region, each private copy starts at the operator's identity (0 for +, 1 for *, etc.), and the copies
    are combined at the end. Operators: `+ - * & | ^ && ||` and `min max`.
- **`#pragma omp parallel for`** combines the parallel region with loop work-sharing. The loop index is private
  automatically. **Canonical loops only:** bounds known before the loop starts, constant increment, the index not
  modified in the body, and no `break`, `return` or `while`/`do` loops. "Error: break statement used with OpenMP
  for loop."
- **OpenMP doesn't check dependences.** A loop-carried dependence (Fibonacci; bubble sort's inner loop) silently
  gives wrong results. Some can be removed by restructuring: the π series with alternating signs needs
  `reduction(+:sum) private(sign)`, or the sign computed from k. Bubble sort becomes **odd-even transposition
  sort** (even phase compares pairs (0,1),(2,3),…; odd phase compares (1,2),(3,4),…). Use `#pragma omp parallel`
  once outside the phase loop and `#pragma omp for` inside, so threads aren't re-forked every phase.
- **Scoping:** always use `default(none)` and state every variable.
  - `shared`: one copy, visible to all.
  - `private`: one copy per thread, uninitialised on entry and undefined on exit.
  - `firstprivate`: private, initialised from the original value.
  - `lastprivate`: copies the value from the sequentially last iteration out of the region.
  - `threadprivate`: a persistent per-thread global that survives across parallel regions (needs
    `omp_set_dynamic(0)` and the same thread count; useful for big per-thread scratch arrays such as derivative
    buffers), initialised from the master with `copyin`.
  - `single copyprivate(x)`: broadcasts a value computed by one thread to all.
  - `default(shared)` or `default(private)` reduce clutter.
- **Scheduling** with `schedule(type[, chunk])`:
  - `static`: iterations assigned before execution, round-robin in chunks. The default is about n/p contiguous
    iterations per thread. With 12 iterations and 4 threads, `static,1` gives t0: 0,4,8. Lowest overhead; best for
    uniform work.
  - `dynamic`: threads grab chunks at runtime. Better load balance, higher overhead.
  - `guided`: large chunks first, shrinking towards `chunk`.
  - `runtime`: taken from `OMP_SCHEDULE`. `auto`: the compiler or runtime decides.
  - Large chunks mean low overhead; small chunks mean better balance.
- **Miscellaneous:** `if(N > 1000)` for conditional parallelism (parallel regions have overhead); `nowait` removes
  the implicit barrier of a work-sharing construct (only when there's no dependence).
- **Synchronisation:**
  - `master` (only thread 0, **no** barrier) and `single` (one thread, with an implicit barrier);
  - `critical` (mutual exclusion; named critical sections are independent);
  - `barrier` (all threads, in the same order, or deadlock);
  - `atomic` (a single update `x op= expr`, `x++`; cheaper than critical; the variable appears once);
  - `flush` (makes a thread's view of shared variables consistent with memory; costly).
- **Task-level parallelism:** `sections`/`section` (each block runs on one thread; extra threads idle; leftover
  sections are handed out dynamically). Modern OpenMP adds `task`/`taskwait`/`taskloop`, `simd`, and `target`
  offload.
- **Runtime:** `omp_set_dynamic`, `omp_set_nested` (`OMP_NESTED`; now `omp_set_max_active_levels`),
  `omp_get_wtime()` (wall-clock seconds) and `omp_get_wtick()` (timer resolution).
- **Performance hygiene** (from my project): pin threads (`OMP_PLACES=cores`, `OMP_PROC_BIND=close|spread`), use
  **first-touch** NUMA placement (initialise data in parallel with the same schedule as the compute loop), avoid
  false sharing (pad per-thread accumulators to 64 B), and verify placement with `OMP_DISPLAY_AFFINITY=true`.

## 4. MPI (distributed memory)

- Interconnects: InfiniBand and Omni-Path at about 100 Gb/s. MPI is a **library** (MPI-4.1; OpenMPI, MPICH,
  MVAPICH, Intel MPI) for C/C++/Fortran. Parallelism is **process-based** in the **SPMD** model, with no incremental
  parallelisation. Compile with `mpicc`/`mpif90` (wrapper scripts) and run with `mpiexec -n 4 ./prog` (or `mpirun -np`).
- **Skeleton:** `MPI_Init`, `MPI_Comm_size(MPI_COMM_WORLD, &p)`, `MPI_Comm_rank(…, &my_rank)`, …, `MPI_Finalize`.
- **Point-to-point:** `MPI_Send(buf, count, type, dest, tag, comm)` and
  `MPI_Recv(buf, count, type, src, tag, comm, &status)`.
  - Messages match on communicator, tag and source/destination; types and sizes should be compatible (the receive
    buffer at least as large as the message).
  - Wildcards `MPI_ANY_SOURCE`/`MPI_ANY_TAG` are for **receivers only**. Find the actual source and tag from
    `status.MPI_SOURCE`/`MPI_TAG`, and the count from `MPI_Get_count(&status, type, &count)`. Use
    `MPI_STATUS_IGNORE` when you don't need them.
  - **Semantics:** the sender builds an *envelope* (source, destination, tag, communicator, size). `MPI_Send` may
    **buffer** (small messages, below an implementation cutoff) or **block** until a matching receive starts;
    `MPI_Recv` always blocks. Messages between a given pair of processes are **non-overtaking**; there's no
    ordering across different senders.
  - **Hangs** come from mismatched tags or ranks (a receive waiting forever), or from every rank calling a blocking
    `Send` first in a ring. That's **unsafe**: it works only while messages fit the system buffer. Test with
    `MPI_Ssend` (synchronous: it never buffers).
  - Safe ring and exchange patterns: alternate send/receive order by rank parity, use **`MPI_Sendrecv`**, use
    `MPI_Sendrecv_replace` (one buffer), or use non-blocking `MPI_Isend/MPI_Irecv` plus `MPI_Wait`/`MPI_Waitall`
    (which also allows overlapping communication with computation).
- **Trapezoid with MPI:** rank r integrates `[a + r·local_n·h, …]`, the non-zero ranks send `local_int` to 0, and 0
  sums. Tree-structured sums do log₂p rounds instead of p−1 receives at the root (3 instead of 7 for p=8, 10 instead of
  1023 for p=1024). Hence collectives.
- **Collectives** match on **communicator and call order** (no tags). Every process must call them in the same order.
  - `MPI_Reduce(send, recv, count, type, op, root, comm)` with `MPI_SUM, MAX, MIN, PROD, LAND, LOR, BAND, BOR, MAXLOC,
    MINLOC`. **Don't alias** the send and receive buffers (`MPI_Reduce(&x, &x, …)` is illegal; use `MPI_IN_PLACE`).
  - `MPI_Allreduce`: everyone gets the result. It's implemented as reduce+broadcast (tree and reverse tree) or a
    **butterfly**.
  - `MPI_Bcast` (tree-structured), `MPI_Scatter`/`MPI_Gather` (block distribution of arrays, e.g. matrix-vector
    products with row blocks), `MPI_Allgather`.
  - The v-variants (`Scatterv/Gatherv/Allgatherv` with counts and displacements, for uneven decompositions),
    `MPI_Alltoall(v)` (transpose and personalised exchange), `MPI_Barrier`.
- **Derived datatypes** let you send non-contiguous or mixed data in one message:
  - `MPI_Type_create_struct(count, blocklens, displs, types, &newtype)` with displacements from
    `MPI_Get_address` (relative to the first element), then `MPI_Type_commit` … `MPI_Type_free`;
  - `MPI_Type_contiguous`;
  - **`MPI_Type_vector(count, blocklength, stride, oldtype, &t)`**, which sends a matrix *column* or sub-block with
    one call (e.g. `MPI_Send(&x[1][1], 1, t, …)`). This is useful for halo exchange in 2D.
  - The alternative is `MPI_Pack`/`MPI_Unpack`.
- **Parallel odd-even sort:** sort locally, then in p phases exchange blocks with the phase partner. The lower rank
  keeps the smaller half and the higher rank keeps the larger half. The exchange must be made safe with
  `MPI_Sendrecv`.
- **Virtual topologies:** `MPI_Cart_create(comm, ndims, dims, periods, reorder, &cart)`, `MPI_Cart_coords`,
  `MPI_Cart_rank`, and **`MPI_Cart_shift(cart, dim, disp, &src, &dst)`**, which gives neighbour ranks for stencil
  codes (`MPI_PROC_NULL` at non-periodic edges, which makes boundary handling automatic).
- **Other calls:** `MPI_Probe` (inspect a pending message's size before allocating), `MPI_Wtime`/`MPI_Wtick`.
- **Halo (ghost) exchange** is the core pattern of every stencil code in this course (A2). Each rank owns a block
  plus `w` ghost cells per side, where w is the stencil reach: 2 on the left and 1 on the right for QUICK; 1 per side
  for the 5-point Laplacian. Before each sweep it exchanges boundaries with `Sendrecv`, and checks global
  convergence with `MPI_Allreduce(MPI_MAX)`.

## 5. OpenACC (directive-based accelerators)

- The family tree: OpenMP (shared memory), Pthreads, MPI and MapReduce (distributed), CUDA and OpenCL (GPU), and
  **OpenACC**: a directive-based, host-directed model for accelerators. The **host** allocates device memory,
  starts transfers, launches parallel, kernels and serial regions, and can run **asynchronously** with the device,
  queueing work and waiting on it. Compile with the NVIDIA HPC SDK: `nvc -acc` (formerly `pgcc`), and
  `-Minfo=accel` to see what the compiler did. Without `-acc` the pragmas are ignored and the code runs serially.
- **Three levels of parallelism:** **gang** (≈ thread block), **worker** (≈ warp), **vector** (≈ threads in a warp).
  Set them with `num_gangs(n)`, `num_workers(n)`, `vector_length(n)` on the region and
  `#pragma acc loop gang | worker | vector` on loops.
- **Compute constructs:**
  - `parallel`: the programmer asserts the parallelism. Gangs are created and *each gang executes the region*
    (redundantly) until a `loop` work-shares it. The gang, worker and vector counts are fixed for the region.
  - `kernels`: the compiler analyses the region, splits it into kernels (usually one per loop nest), and only
    parallelises what it can prove safe. Kernels run in order with a barrier at the end unless `async`.
  - `serial`: a single gang, worker and lane.
  - `if(cond)` and `async`/`wait`.
- **The key pitfall:** `parallel loop` trusts you. On a loop with a backward dependence (`a[i] = a[i-1] + …`) the
  compiler still parallelises it and produces garbage (the slide's output: 32 32 0 0 …). `parallel loop auto` or
  `kernels` analyse it and fall back to `seq` ("Loop carried dependence of a prevents parallelization"), giving the
  correct result.
- **Data management** (where most of the performance is):
  - Without directives, each compute region does **implicit** `copyin`/`copyout`/`copy` (shown in `-Minfo` as
    "Generating implicit copyout(a[:])"), which means repeated transfers.
  - Structured `#pragma acc data copyin(A,B) copyout(C) { … }` spans several kernels. Inside, use `present(A,B,C)`
    to assert the data is already on the device.
  - Clauses: `copy` (in and out), `copyin`, `copyout`, `create` (device only, no copy), `delete`, and `present`.
    Array sections are `a[start:length]` in C and `a(start:end)` in Fortran.
  - **Unstructured** `#pragma acc enter data copyin(…) create(…)` … `exit data copyout(…) delete(…)` can span
    functions (one exit per enter).
  - `#pragma acc update self(a[0:n])` makes the host match the device; `update device(…)` makes the device match
    the host. Use these in the middle of a data region.
  - **Shallow copy:** a pointer inside a struct is copied as an address, not as the pointed-to object (a deep-copy
    homework).
  - The PGI profiler output ("data region reached 4 times; copyin transfers: 2…") is how you verify a data
    optimisation.
- **Loops:**
  - Nested `parallel loop` inside `parallel loop` isn't supported. By default a `loop` applies to the next loop
    only, and inner loops run `seq`.
  - `collapse(n)` merges a nest into one iteration space; `seq`, `independent` (an assertion) and `auto` control
    parallelisation.
  - `reduction(op: vars)` works like OpenMP (`+ * max min & | ^ && ||`).
  - `tile(…)`, `gang/worker/vector` mapping.
  - `routine seq` marks functions called inside kernels.
- **Case studies:**
  - **Caesar cipher:** a pointer-walking `while (*iptr) *optr++ = *iptr++ + offset;` loop isn't parallelisable,
    because it has no integer trip count and its loop-carried pointer dependences make the compiler fall back to
    sequential. Worse, the missing data clauses caused "Accelerator Fatal Error: call to cuStreamSynchronize
    returned error". Fix: rewrite it as an indexed `for` loop with an explicit `copyin(input[0:N])
    copyout(output[0:N])`.
  - **Matrix addition** optimised step by step: implicit copies, then a `data` region, then `present`, then
    `collapse(2)` (which turns "gang + vector(96)" with an inner `seq` loop into a collapsed gang-vector loop).

## 6. Performance analysis (used throughout and in A3's hand calculation)

```
T(n,1) = σ(n) + φ(n)                        σ: inherently serial part, φ: parallelisable part
T(n,p) = σ(n) + φ(n)/p + κ(n,p)             κ: communication/synchronisation overhead
ψ(n,p) = T(n,1)/T(n,p)  (speedup)           ε = ψ/p  (efficiency)
Amdahl (fixed n):     ψ ≤ 1 / (f + (1−f)/p),   f = σ/(σ+φ),  ψ_max = 1/f
Gustafson (scaled):   ψ = p − (1−f)(p − 1)   [f here = serial fraction of the parallel run]
Karp–Flatt metric:    e = (1/ψ − 1/p) / (1 − 1/p)   (experimentally determined serial fraction;
                      growing e with p ⇒ overhead, not just serial code, is limiting you)
Strong scaling: fixed problem, more workers.   Weak scaling: problem ∝ workers (ideal: constant time).
```

A3's numbers: n=1000, p=100 gives σ = 11,000, φ = 1,300,000, κ = 3·(1000·log₁₀100 + 100) = 6,300,
T(n,p) = 30,300, ψ = 43.27, ε = 43.3%, f = 11/1311 ≈ 0.84%, Amdahl ψ(100) = 54.6, and ψ_max = 119.2. The gap
between 43.3 and 54.6 is entirely the communication term that Amdahl ignores.

## 7. Tutorials (Jan–May 2026)

| # | Content |
|---|---|
| 1 | Linux, C/Fortran basics, gnuplot; show `A + Aᵀ` is symmetric; time a matrix-vector product for N = 64…512 |
| 2 | Install OpenMP; hello world with and without OpenMP; print in rank order; **serial trapezoid** for ∫₁^π sin x/(2x³) dx (exact 0.198557, n = 32) |
| 3 | `parallel for` and reduction: **Simpson's rule** (much smaller error than trapezoid), parallel matrix addition (p = 2, 4, 8), **sieve of Eratosthenes** (serial and an attempted OpenMP version) |
| 4 | Time OpenMP matrix add/multiply; first-derivative FD schemes (1st, 2nd, 4th order) for u = 7 − x tan x on [−1, 1] with Δx = 0.01, 0.001; install MPI |
| 5 | MPI hello; root sums a float from every rank; MPI trapezoid; how to parallelise the sieve with MPI |
| 6 | Collectives (Reduce, Allreduce, Scatter, Gather, Allgather); **row-block matrix-vector product** with collectives; `MPI_Type_create_struct` broadcast |
| 7 | MPI derivatives of x³ − sin 5x on [0, 3] with **ghost points**, 1st/2nd/4th-order schemes, error vs Δx; Gatherv/Scatterv/Alltoall/Wtime; install the NVIDIA HPC SDK |
| 8 | OpenACC: gangs and `async` behaviour; fix `break` inside a parallel loop (find a number); more debugging exercises |
| 9 | OpenACC debugging: infection-percentage code (illegal `rand()` in device code, a wrong `collapse`/`reduction`, missing data clauses), OpenACC matrix multiply, **Padé + TDMA** (`q3_pade_tdma.c`) |
| 10 | OpenACC **QR by modified Gram–Schmidt** (M = 1500, N = 1000; data-transfer optimisation) and a 1D **convolution filter** (K = 5) |

## 8. My assignments

### 8.1 A1 (Mar 2026): OpenMP triple integral and Poisson–Jacobi

- **Q1, triple integral** `∫₀⁴∫₀³∫₀² (4x³ + xy² + 5y + yz + 6z) dz dy dx = 2040` (hand-derived in
  `hand_calculations.md`).
  - (a) **Midpoint Riemann sum** with Nx, Ny, Nz = 48, 36, 24 (Δ = 1/12) gives 2039.639 (0.018% error). Uses
    `#pragma omp parallel for collapse(3) reduction(+:result)`, and serial and parallel agree to about 1e-12.
    Speedup is < 1 (41K evaluations take 0.09 ms, so thread start-up dominates), which is correctly explained.
  - (b) **Monte Carlo** `I = V·f̄ ± V·f_rms`, using thread-safe `rand_r` with per-thread seeds. The error falls as
    1/√N, as verified (213 at N=100 down to 1.84 at N=10⁶), and it reaches 7.08× on 8 threads at N = 10⁵.
- **Q2, Poisson** `∇²φ = −2(2 − x² − y²)` on [−1, 1]² with φ = 0 on the boundary. The exact solution is
  `φ = (x²−1)(y²−1)`, verified. Jacobi stops when within 1% of the exact solution, falling back to absolute error
  where exact ≈ 0 (good). The parallel version uses **one persistent parallel region** with **parity
  double-buffering** (`rd = iter%2`), so there's only one barrier per iteration. Iterations: 400 at Δ = 0.1, 37,900
  at Δ = 0.01, 151,400 at Δ = 0.005, matching O(1/Δ²). Speedups were 3.29× and 3.52× with 8 threads (this laptop
  has 4 cores and 8 hyperthreads, so ~3.5× is about right). For this manufactured solution, the 5-point stencil is
  *exact* (φ is quadratic in each variable, so the truncation error vanishes), so stopping at 1% of exact is
  well-posed.
- **Verdict:** solid and correct. Small improvements: report the hardware (cores and HT); add a Karp–Flatt or
  efficiency plot.

### 8.2 A2 (27 Mar → 10 Apr 2026): MPI advection and MPI Poisson ⚠️

- **Q1, 1D advection** `u_t + c u_x = 0`, c = 1, L = 2, u₀ = sin 4πx on [0, 0.5]. Euler explicit with Δx = 0.002 and
  Δt = 1e-4 (CFL = 0.05). Two schemes: **first-order upwind** `u_i − CFL(u_i − u_{i−1})`, and **QUICK**
  `u_i − CFL(3/8 u_i − 7/8 u_{i−1} + 1/8 u_{i−2} + 3/8 u_{i+1})`, falling back to upwind at i = 1.
  - The MPI version uses block decomposition with **2 left and 1 right halo** cells exchanged by `MPI_Sendrecv`
    each step, then `Gatherv` to rank 0. Serial and MPI (p = 2, 4) match **bit for bit**.
  - The analysis is correct: upwind is **diffusive** (its leading error term ∝ Δx·u_xx acts like artificial
    viscosity; the peak drops to about 0.85 by t = 1). QUICK keeps the amplitude but adds **dispersive wiggles**
    at the slope discontinuities (x = 0 and 0.5). That's the classic dissipative-versus-dispersive trade-off.
- **Q2, Poisson** `∇²φ = −(x² + y²)` on [−1, 1]². φ = 0 at y = ±1; φ = sin 2πy at x = −1; **Neumann ∂φ/∂x = 0
  at x = 1** via the 2nd-order one-sided formula `φ_N = (4φ_{N−1} − φ_{N−2})/3`. Jacobi stops when
  `max|φ^{k+1} − φ^k| < 1e-4` (as the assignment prescribes). (a) Serial at Δ = 0.1 takes 387 iterations.
  (b) MPI at Δ = 0.01 with row-block decomposition, one ghost row each side, `Sendrecv`, `Allreduce(MAX)` and
  `Gatherv`, takes **1013 iterations**, identical to serial for p = 2, 4, 8.
- **Problem I found: the Δ = 0.01 solution is nowhere near converged.** I re-ran my own `q2_serial.c` to a 1e-11
  tolerance. The true solution needs **188,163 iterations** (12.5 s). Comparing the profiles:

| Profile | at 1013 iterations ("converged" in the report) | truly converged |
|---|---|---|
| φ(x, y=0) at x = 0.6 | 0.010 | 0.226 |
| φ(x, y=0) at x = 1.0 | 0.021 | 0.264 |
| φ(0, y) at y = 0 | ≈ 0.002 | 0.145 |

  The plotted Δ = 0.01 curves are **wrong by about 90–99%**. Serial and MPI still "match exactly" because both stop
  at the same unconverged iterate. The cause is the rule from §2.5: with `ρ_J ≈ 1 − π²Δ²/2 ≈ 1 − 1.2e-4`, a 1e-4
  successive change means the error is still O(1). The report's explanation is also wrong: "iterations grow ∝ N,
  characteristic of the spectral radius cos(π/N)". The spectral radius predicts **O(N²)** growth. The observed
  linear growth is an artefact of the weak criterion.
- **Phase-2 fix:** use a residual-based or tolerance-scaled criterion (`‖r‖/‖b‖ < 1e-8`, or `tol·(1−ρ)`), report
  both the prescribed-criterion count and the true convergence, show the Δ = 0.1 vs Δ = 0.01 difference, and
  optionally add Gauss–Seidel red-black or SOR to show the iteration savings.

### 8.3 A3 (17 → 27 Apr 2026): hand calculations and OpenACC Cholesky

- **Hand calculation 1 (speedup, efficiency, Amdahl):** correct (see §6).
- **Hand calculation 2 (recursive doubling for the 4×4 system):** my answer x = (1,1,1,1) is right, but **the method
  I showed is cyclic reduction**: it eliminates the odd unknowns, halves the system, then back-substitutes. I labelled
  it "Recursive doubling (cyclic reduction)". The question explicitly said: *"You will get points only if you present
  your solution using recursive-doubling algorithm that we learnt in the class."* In the course's (and my own notes')
  version of recursive doubling, every equation is kept, the stride doubles, and the matrix becomes diagonal. That
  likely cost marks. The correct working is in §2.4.
- **Programming question: Cholesky with OpenACC** (`A = LLᵀ`, SPD matrix, in place).
  - First I fixed the given code's bug: it calls `init()`, which doesn't exist; the function is `initmult()`. (The
    spec's formula also has a typo, `L(k,j)` instead of `L(j,k)`; my code uses the correct one.)
  - **V1, column-oriented** (2N kernel launches instead of O(N²/2)): one structured `data copy(a[0:n*n])` region.
    The diagonal reduction runs on the device (`parallel loop reduction(+:s) present(a)`). The host computes the
    square root using its untouched copy of `A(jj,jj)` (valid, because Step B only writes below-diagonal entries),
    then `update device(a[jj*n+jj:1])`. A gang-vector loop over rows computes the column below the diagonal
    (`num_gangs` 32 or 256, `vector_length(128)`).
  - **V2, blocked**: POTRF on an NB=64 diagonal block (serial on the device), then TRSM, then a `collapse(2)` SYRK
    trailing update.
  - Verified against serial with max difference below 1e-7, and the N = 10 L matrix printed matches.
  - **Timing** (MX130, a laptop Maxwell GPU) for N = 10, 100, 1000, 2000, 5000: serial 2e-6, 4.6e-5, 0.025, 0.29,
    10.27 s. V1: 0.094, 0.0058, 0.15, 0.90, 12.27 s. V2: 7.5e-5, 0.0032, 0.089, 0.50, **7.98 s**. **The GPU only wins
    at N = 5000, and only by 1.29×.**
  - **Why it's slow:** in Step B, thread `ii` reads `a[ii*n + kk]`, so consecutive threads touch addresses n apart:
    **completely uncoalesced**. There's also a host round-trip per column, a serial POTRF block, and a weak GPU.
- **Phase-2 improvements:** store Lᵀ, or use a column-major copy, so threads read consecutive addresses; do the
  diagonal on the device to remove the per-column host sync; use a left-looking blocked algorithm with a GEMM-like
  update; compare against cuSOLVER `potrf` as the roofline reference.

## 9. Final project: parallel ray tracing for analytically defined objects with point classification

### 9.1 What I built (Apr–May 2026)

A from-scratch **C++17** renderer with no external dependencies: about 4,000 lines across `src/`, `include/`,
`tests/` and `tools/`. It was built with an AI-assisted workflow (`CLAUDE.md` sets the design invariants), and
there's a separate `submission/` copy.

- **Geometry kernel:** rays `o + td` with a cached `1/d`. Analytical intersection for **sphere** (normalised
  quadratic `b'² − ac`), **plane** (optionally clipped, for the Cornell walls), **axis-aligned box** (slab method),
  and **finite cylinder** (side quadratic plus two cap discs). Normals face the incoming ray, with a front-face flag
  used for Fresnel transitions.
- **Point classification** (the problem statement's second half): the sign of the implicit function `f(p)` with
  tolerance ε gives inside, on or outside. It propagates through CSG with Boolean rules
  (`A∪B = A∨B`, `A∩B = A∧B`, `A\B = A∧¬B`), and "on" propagates conservatively.
- **CSG by ray-interval merging** (Algorithm 1): each primitive reports the t-intervals where the ray is inside it.
  A **sweep line** over the sorted interval endpoints tracks depth counters dA and dB and evaluates the operator's
  predicate. Tie-breaks merge touching intervals and avoid zero-length overlaps; normals from B are negated in a
  difference. Scenes include swiss cheese (a box minus 27 spheres), a Menger sponge, a lattice (Steinmetz bicylinder,
  lens stacks), and nesting 12 levels deep.
- **BVH with binned SAH** (12 bins, C_t = 1, C_i = 1.5, a median-split fallback). Unbounded planes are kept outside
  the BVH.
- **Integrators:** a **Whitted** recursive shader (Blinn–Phong, hard shadows, mirror reflection, dielectric with
  Schlick Fresnel, Snell's law and total internal reflection). And a **Monte Carlo path tracer**: iterative
  throughput β, **next-event estimation** to parallelogram area lights, **MIS** with the power heuristic (Veach),
  cosine-weighted hemisphere sampling (Malley), **Russian roulette** from bounce 5, stratified √spp × √spp sampling
  with **Cranley–Patterson rotation**, and **PCG32** seeded per pixel and sample: `seed = yW + x + 1000003·s`.
- **Determinism by construction:** seeds depend only on (pixel, sample), and per-pixel accumulation happens in a
  thread-private accumulator before one store. So the output is **byte-identical** across thread and rank counts.
  The MD5 oracles are `a45169cd…` (Whitted showcase) and `89597491…` (Cornell 1024²×256 spp), verified for OpenMP
  T = 1…40 and MPI np = 1…160.
- **Parallelisation:**
  - OpenMP over **32×32 tiles** with `schedule(dynamic,1)`.
  - MPI **row-strip** tile ownership with two `MPI_Gatherv` calls (pixels plus tile metadata) to rank 0.
  - **Hybrid** with one rank per socket (`OMP_NUM_THREADS=20 OMP_PLACES=cores OMP_PROC_BIND=close mpirun -np 2N
    --map-by ppr:2:node`).
  - **First-touch**: the pixel buffer is zeroed in parallel.
  - An optional **adaptive LPT** pre-pass: a cheap preview estimates per-tile cost, then tiles are scheduled
    longest-first.
- **Infrastructure:** unit tests (primitives, classification, CSG, BVH, path), `tools/bench.py` (pinned runs, CSV),
  `aggregate.py`, `merge_results.sh`, `verify_run.sh` (MD5 checks), `diff_ppm.py`, and PBS job scripts with an
  `AQUA_RUNBOOK.md`. Cross-renderer comparison with **Mitsuba 3** (scalar and LLVM); pbrt-v4 was deferred after its
  build failed on AQUA.
- **Report:** ASME two-column format, 8 pages, with a rendering-pipeline flowchart and algorithms.

### 9.2 Results as reported (AQUA: Xeon Gold 6248, 2 × 20 cores per node, 4 nodes, IB 100 Gb/s)

- **OpenMP strong scaling** (Cornell 1024² × 256 spp): T = 1 takes 378.3 s; T = 2, 2.01×; 4, 3.93×; 8, 7.19×;
  16, 11.02×; 20, 12.16×; **40, 38.96× (97.4%)**. At 1024 spp, T = 40 renders in 38.3 s.
- **Hybrid:** np = 4 × t = 20 on 2 nodes, **65.3×** (81.6% efficiency on 80 cores). On a single node pure OpenMP
  wins.
- The Cornell box shows correct **colour bleeding** (qualitatively matching a Mitsuba 8192-spp reference).
  RMSE, SSIM and FLIP were deferred.

### 9.3 Critical review (what a reviewer or interviewer will ask)

1. **The "single-socket NUMA" explanation of the T = 16/20 dip can't be right.** At T = 40 each socket still runs
   20 threads, so per-socket L3 and memory contention is *the same* as at T = 20, yet efficiency "recovers" from
   61% to 97%. A Cornell box's working set (tens of primitives, a tiny BVH, per-tile pixel writes) also fits in L1
   and L2. The dip **reproduces on two nodes (cn155, cn186) in two separate runs**. The T = 16/20 timings (34 s,
   31 s) sit about 1.5× above the trend from T ≤ 8, while T = 40 is on trend. The likely cause is **thread
   placement**: hyperthread sibling enumeration, or interleaved socket numbering on AQUA with
   `OMP_PLACES=cores`/`close`. To check: `OMP_DISPLAY_AFFINITY=true`, `lscpu -e`, `numactl -H`, explicit
   `OMP_PLACES="{0}:20"`, `perf stat` for frequency. Fix the explanation before publishing anything.
2. **Pure-MPI scaling is broken, and that's not inherent.** From the raw `results/merged/timings.csv`: np = 1:
   385 s; np = 2: 204 s; np = 4: 225 s; np = 8: 230 s; np = 16: 232 s; np = 32: 235 s; np = 40–160: about 295 s
   (only 1.3×). A 4-way strip split can't have a *worse* maximum share than the 2-way split it refines, so this
   isn't load imbalance or the Gatherv cost (3 MB is milliseconds over IB). The same signature appears in
   **np = 1 × t = 40 under mpirun at 385 s** (40 threads, no speedup) and np = 2 × t = 20 at 204 s. **Ranks are
   pinned to single cores** (OpenMPI's default `--bind-to core` for np ≤ 2, and probably a mapping or `-ppn`
   leftover for the pure-MPI sweep). The report says "pure-MPI scaling is not the use case… bound by MPI_Gatherv",
   which misattributes a binding bug. Fix: `--bind-to none` or `--map-by ppr:…:socket:pe=T`, and
   `--report-bindings` to confirm.
3. **The "novel" adaptive LPT scheduler is a negative result** in my own data. It's slower than
   `schedule(dynamic,1)` in most cells (Whitted T = 8: 251 vs 201 ms; T = 20: 180 vs 149 ms; swiss cheese T = 8: 91 vs
   54 ms; Cornell T = 40: 10.0 vs 9.8 s). Fine-grained dynamic scheduling already balances 1024 tiles, so a preview
   pass is pure overhead. Report it honestly as a negative result.
4. **Number hygiene:** the report's Table 5 hybrid np=4×t=20 says 5,795 ms, but the merged dataset median is
   6,327 ms. The headline 38.96× vs 38.87× differs depending on which T = 40 rows are included. Regenerate every
   table from one script.
5. **Overclaims to avoid:** `CLAUDE.local.md` says "6–10× faster than pbrt-v4". That compares analytic primitives
   with triangle meshes, different samplers, and pbrt RMSE 0.137 (non-equivalent scenes). "No renderer combines
   analytical primitives + CSG + MPI" is a niche claim, not a contribution by itself. Quantitative image quality
   (RMSE, SSIM, FLIP versus the GT EXR, already captured) is the missing validation.
6. **Genuinely strong parts:** the reproducibility discipline (bit identity across parallel configurations), the
   correct CSG interval logic, a proper path tracer (NEE + MIS + RR + stratification), good OpenMP scaling at
   T = 40, a disciplined benchmark pipeline, and honest documentation of what failed (pbrt build, MPI rescues).

**Phase-2 plan for the project:** fix MPI binding and re-measure strong scaling to 160 cores; resolve the T = 16/20
anomaly; replace row strips with **dynamic tile distribution across ranks** (master–worker, or work-stealing with
`MPI_Get`/RMA); compute RMSE/SSIM/FLIP time-to-quality curves; add SIMD packet traversal (AVX2) of the analytic
intersections. Then it's a solid workshop paper (EGPGV short paper) or a strong resume item: *"Hybrid MPI+OpenMP
path tracer with CSG; bit-reproducible across 160 cores; 39× on 40 cores, 65× across nodes."*

## 10. Revision questions

1. **Why does `reduction` beat `critical` for sums?** Private partial sums plus one combine, instead of
   serialising every update.
2. **`private` vs `firstprivate` vs `lastprivate` vs `threadprivate`?** Uninitialised; initialised from the
   original; copies out the last iteration's value; persistent per-thread global across regions.
3. **When does `static` beat `dynamic` scheduling?** For uniform per-iteration work, because the overhead is lower.
   Use dynamic or guided for irregular work (ray tracing tiles).
4. **Why is an `MPI_Send`/`MPI_Send` ring unsafe, and what fixes it?** It relies on system buffering. Fix with
   `Sendrecv`, parity ordering, non-blocking calls, or test with `Ssend`.
5. **Collectives vs point-to-point matching?** Collectives match on communicator and call order; point-to-point
   on communicator, tag and source.
6. **How do you send a column of a C matrix in one message?** `MPI_Type_vector(nrows, 1, ncols, MPI_DOUBLE, &t)`.
7. **Halo width for QUICK?** 2 cells upstream, 1 downstream.
8. **OpenACC `parallel` vs `kernels`?** The programmer asserts independence, versus the compiler proving it.
   `parallel loop` silently parallelises dependent loops.
9. **How do you avoid repeated host-device copies in OpenACC?** A `data` region or `enter data` + `present`, and
   `update` only when needed.
10. **TDMA vs recursive doubling vs cyclic reduction?** O(n) sequential; O(n log n) with every equation parallel,
    ending diagonal; O(n) with half the equations per level plus back-substitution.
11. **Why do Jacobi iterations scale as O(N²)?** ρ = cos(πΔ) ≈ 1 − π²Δ²/2, so you need about 2/(π²Δ²) iterations
    per decade of error reduction.
12. **Why is a successive-difference stopping test unreliable?** True error ≈ tol/(1−ρ), which is huge when
    ρ → 1. Use residuals.
13. **Upwind vs QUICK?** First order and dissipative (smears) versus third order and dispersive (wiggles near
    discontinuities).
14. **Strong vs weak scaling; Karp–Flatt?** Fixed vs scaled problem size. The Karp–Flatt metric gives the
    experimentally determined serial fraction.
15. **What's first-touch NUMA placement?** Pages land on the NUMA node of the thread that first writes them, so
    initialise data in parallel with the compute schedule.

## 11. Bridges to AI infrastructure

| ID5130 concept | AI-systems counterpart |
|---|---|
| MPI collectives (Reduce, Allreduce via tree or butterfly, Allgather, Alltoall(v)) | **The exact API of distributed training**: data-parallel gradient `all_reduce` (ring and tree in NCCL), FSDP/ZeRO `all_gather` + `reduce_scatter`, MoE expert parallelism `all_to_all(v)`, tensor parallelism `all_reduce` per layer. `torch.distributed` mirrors MPI semantics (ranks, communicators = process groups, collectives) |
| Domain decomposition + halo exchange | Tensor and sequence parallelism (sharding activations with boundary communication), context parallelism and ring attention (KV blocks passed around a ring, like halo exchange) |
| Hybrid MPI+OpenMP, one rank per socket, NUMA, first-touch | Multi-node training layout: one process per GPU, NVLink within a node, IB across nodes; CPU data loaders pinned per NUMA node; topology-aware placement |
| Load balancing (static vs dynamic, LPT) | Batching variable-length sequences (packing, bucketing), MoE expert load balancing, pipeline-stage balancing; why dynamic beats clever static schedules |
| Iterative solvers, convergence and stopping criteria | Optimisation convergence diagnostics; numerical precision (FP32 vs BF16 accumulation) |
| Cholesky, QR, tridiagonal solvers | Second-order and preconditioned optimisers (Shampoo, SOAP use matrix roots and Cholesky); Gaussian processes; low-rank adapters |
| Determinism and bit-reproducibility (my project) | **Reproducible training/inference** is a live problem: non-deterministic atomics and reduction orders in GPU kernels, batch-invariant inference kernels; my per-pixel seeding plus thread-private accumulation is the same idea as deterministic reductions |
| Monte Carlo, variance, MIS | Sampling in diffusion models and RL; importance sampling in LLM RL (PPO/GRPO ratios); variance reduction |
| OpenACC directives ↔ compiler-managed parallelism | High-level kernel compilers (Triton, torch.compile); the "trust the compiler vs assert parallelism" trade-off |

**Project seed:** an **MPI-style collective-communication benchmark and simulator for LLM training**, which reuses
this course directly. Implement ring, tree and halving-doubling all-reduce in MPI, model them with an α–β cost model,
validate on AQUA (CPU) and on GPUs with NCCL, and predict data-parallel training step time for a given model and
cluster. That's systems depth with a direct AI link.

## 12. References

- P. Pacheco, *An Introduction to Parallel Programming*, Morgan Kaufmann, 2011. M. J. Quinn, *Parallel Programming in C
  with MPI and OpenMP*, McGraw-Hill, 2004 (source of the σ/φ/κ speedup model and Karp–Flatt).
- P. Moin, *Fundamentals of Engineering Numerical Analysis*, 2nd ed., CUP, 2010 (finite differences, Padé schemes).
- R. W. Hockney, "A fast direct solution of Poisson's equation using Fourier analysis", JACM 1965; H. S. Stone, "An
  Efficient Parallel Algorithm for the Solution of a Tridiagonal Linear System of Equations", JACM 1973 (recursive
  doubling); B. L. Buzbee, G. H. Golub, C. W. Nielson, "On Direct Methods for Solving Poisson's Equations", SIAM JNA 1970
  (cyclic reduction).
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003.
- B. P. Leonard, "A stable and accurate convective modelling procedure based on quadratic upstream interpolation
  (QUICK)", CMAME 1979.
- OpenMP 5.2 specification; MPI 4.1 standard; OpenACC 3.3 specification; NVIDIA HPC SDK documentation.
- Rendering: T. Whitted, CACM 1980; J. Kajiya, "The Rendering Equation", SIGGRAPH 1986; E. Veach, L. Guibas, MIS,
  SIGGRAPH 1995; I. Wald, binned SAH BVH, IRT 2007; M. Pharr, W. Jakob, G. Humphreys, *PBRT* 4th ed.; W. Jakob et al.,
  Dr.Jit / Mitsuba 3, TOG 2022; M. O'Neill, PCG, 2014; C. Goral et al., the Cornell box, SIGGRAPH 1984.
