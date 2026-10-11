# CS6400 (slides: "CS6640") — Advanced Computer Architecture — Learning Notes

> Instructor: **Prof. Madhu Mutyam**, PACE Lab, Dept. of CSE, IIT Madras. Jan–May 2025.
> One-file revision guide covering all 21 lectures (re-explained lesson by lesson with the background the slides
> assume), both programming assignments (what I built, how I verified it, the bugs, the Phase-2 fixes), revision
> questions, and bridges to AI infrastructure.
>
> **Folder vs. slides:** the folder is named `CS6400_ACA`, but every slide header says *CS6640: Advanced Computer
> Architecture*. The `LectureSlides/LecN.pdf` files are Beamer slides printed six per A4 page and annotated on an
> iPad. My handwritten notes, mostly in Lec11 and Lec12, are folded in below.
>
> **Evaluation (Lec1):** paper presentation 10, technical report 10, mid-sem 40, projects/programming assignments 40.
> A minimum of 40 overall and at least 25% in each component are needed to pass.
> **Texts:** Culler, Singh & Gupta, *Parallel Computer Architecture: A Hardware/Software Approach* (1998);
> Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 7th ed. (2025);
> Nagarajan, Sorin, Hill & Wood, *A Primer on Memory Consistency and Cache Coherence*, 2nd ed. (2020);
> and research papers.

---

## Table of contents

- Part A — Sequential architecture recap: L1 Introduction · L2 RISC-V
- Part B — Parallel computer architecture: L3 Programmability · L4 Parallel memory systems · L5 Coherence ·
  L6 Consistency · L7 Synchronisation & TM · L8 HW support for programming/debugging · L9 Manycore ·
  L10 Interconnection networks · L11 Dark silicon
- Part C — Heterogeneous architecture: L12 Domain-specific architectures & TPU · L13–14 Data-level parallelism &
  GPUs · L15 FPGAs & CPU–FPGA systems · L16 Amdahl's law in the multicore era & big.LITTLE · L17 Consistency and
  coherence for heterogeneous systems
- Part D — Emerging concepts: L18 Processing-in-memory · L19 Approximate computing · L20 Non-volatile memories ·
  L21 Chiplets & HBM
- Part E — My assignments: A2 Snooping MSI/MESI simulator · A3 big.LITTLE scheduler
- Part F — Revision questions · Part G — Bridges to AI infrastructure · Part H — References

The course plan (Lec1) listed four topics that the final "Topics covered" slide (Lec21) strikes out as **not
covered**: debugging heterogeneous systems, ML and inference acceleration, neuromorphic computing, and
graph-centric architectures. 3D stacking was folded into the chiplets/HBM lecture, and CPU/GPU hybrids into the
DLP lecture.

---

# Part A — Sequential architecture recap

## L1 (22 Jan) — Introduction: computing power, the end of the line, "plenty of room at the top"

**Phases of single-processor performance growth** (Hennessy & Patterson, Turing lecture, CACM 2019):

| Era | Growth | Reason it ended |
|---|---|---|
| CISC (≈1978–86) | 2× / 2.5 yr (22%/yr) | microcode complexity |
| RISC (≈1986–2003) | 2× / 1.5 yr (**52%/yr**) | ILP plus Dennard scaling: pipelining, superscalar, OoO, caches |
| End of Dennard scaling → multicore (≈2003–11) | 2× / 3.5 yr (23%/yr) | power wall |
| Amdahl's law limits (≈2011–15) | 2× / 6 yr (12%/yr) | serial fractions |
| "End of the line" (2015–) | 2× / 20 yr (3%/yr) | Moore's law slowing too |

- **Moore's law:** transistor density doubles every two years (Moore's 1975 revision of the 1965 yearly doubling).
- **Dennard scaling** (1974): as transistors shrink by S, voltage and current shrink too, so **power density stays
  constant**. It slowed from about 2007 and was essentially gone by 2012, because threshold and supply voltages
  stopped scaling (leakage).
- **Speculation is wasteful:** on an Intel Core i7, some SPECint benchmarks waste up to about 39% of executed
  instructions on mis-speculation (Turing lecture, Fig. 4). Wider and deeper ILP therefore costs energy, which
  is why ILP stopped being the main lever.
- **"There's plenty of room at the Top"** (Leiserson et al., *Science* 2020). Once "the Bottom" (semiconductor
  scaling) stalls, performance must come from the Top:
  - **Software performance engineering**: remove bloat and tailor software to the hardware. Their 4096×4096
    matrix-multiply example goes from naive Python to parallel, tiled, vectorised C with AVX intrinsics for a
    speed-up of about 60,000×.
  - **Algorithms**: new algorithms, new problem domains, new machine models.
  - **Hardware architecture**: **hardware streamlining**, i.e. simpler processors and domain specialisation.
    This is the motivation for the whole second half of the course (DSAs, GPUs, FPGAs).

**Uniprocessor microarchitectural optimisations recap** (the "Brief overview of sequential architecture"):
bit-level parallelism (wider ALUs: an 8-bit add done as two 4-bit adds with carry, versus one 8-bit add),
instruction pipelining (IF→ID→RD→EX→WB), caches, prefetching, branch prediction, ILP via an out-of-order pipeline
(in-order fetch/decode/dispatch → out-of-order issue/execute/finish via reservation stations → in-order
complete/retire via the reorder buffer and store buffer), and hardware multithreading (several PCs and register
sets sharing one pipeline). See `CS6600_CompArch_Learning.md` for the details.

## L2 (24 Jan) — RISC-V open ISA

**ISA** = the hardware/software contract: the processor's functionality as visible to compilers.
History: single accumulator (EDSAC 1950) → accumulator + index registers (Manchester Mark 1, IBM 700) →
high-level-language machines (B5000) and stack machines (B6500, HP3000) → general-purpose-register machines,
splitting into **CISC register-memory** (VAX, x86) and **load/store register-register** (CDC 6600, Cray-1)
→ **RISC** (MIPS 1985, SPARC, HP-PA, RS/6000, PowerPC).

**RISC-V:** started in 2010 at UC Berkeley (Krste Asanović et al.) as a "3-month project". It is license-free and
royalty-free, and governed by the RISC-V Foundation (2015; now RISC-V International). Design maxims: *simplicity
favours regularity, make the common case fast, smaller is faster.*

- **Naming:** `RV[32|64|128][extensions]`. Base integer ISAs are **RV32I/RV64I** (RV32E has 16 registers for
  embedded). Standard extensions: **M** (mul/div), **A** (atomics: LR/SC and AMOs), **F/D/Q** (FP single, double,
  quad), **C** (16-bit compressed), and **G** = IMAFD. Examples: `RV32IMAC`, `RV64GC`, `RV64GCXext`
  (X = non-standard extension). Also V (vector).
- **Registers and ABI:** 32 integer registers; x0 is hard-wired zero. x1 `ra` and x2 `sp`; x5–7 and x28–31 are
  `t0–t6` temporaries (caller-saved); x8–9 and x18–27 are `s0–s11` saved (callee-saved; s0 doubles as the frame
  pointer); x10–17 are `a0–a7` arguments, with a0–a1 also return values (caller-saved).
- **Operands:** registers, immediates, memory (byte-addressable, little-endian, load/store architecture only).
- **Large constants:** `lui rd, upper20` then `addi rd, rd, lower12`. `addi` **sign-extends** its 12-bit
  immediate, so if bit 11 of the low part is 1, the upper part must be incremented by 1.
  For `0xFEDC8765`: `lui s0,0xFEDC8; addi s0,s0,0x765`. For `0xFEDC8E65` the low part 0xE65 is negative as a 12-bit
  value (0xE65 − 0x1000 = **−411**), so `lui s0,0xFEDC9; addi s0,s0,-411`. (The slide writes −341, which
  reconstructs 0xFEDC8EAB, so that's a typo.)
- **Addressing modes:** register only (`add`), immediate (`ori t2,t3,0xFF`), base + displacement
  (`lw s4,20(zero)`), and PC-relative (branches and `jal`; offsets are in bytes and must be even).
- **Instruction formats** (all 32-bit, aligned; rd, rs1 and rs2 always in the same bit positions; the immediate
  sign bit is always bit 31, which makes sign extension fast):
  **R** (reg-reg), **I** (immediates, loads, `jalr`), **S** (stores), **B** (branches; immediate in units of
  2 bytes), **U** (`lui`, `auipc`), **J** (`jal`; 20-bit offset).
- **Control transfer:** `jal rd, label` (J-type; rd = PC+4), `jalr rd, imm(rs1)` (I-type; indirect jumps and
  returns), and conditional branches `beq, bne, blt, bge, bltu, bgeu` comparing rs1 and rs2.
- **Toolchain:** GCC, LLVM, Spike (ISA simulator), QEMU; OS support from Linux and FreeRTOS.

---

# Part B — Parallel computer architecture

## L3 (29 Jan) — Parallel computer architecture: programmability

- Many slower cores instead of one big core (the "elephant vs. 4 horses" picture): power grows super-linearly with
  single-core performance, so several simpler cores give more throughput per watt.
- **Definition** (Almasi & Gottlieb): *a parallel computer is a collection of processing elements that can
  communicate and cooperate to solve a large problem fast.*
  **PCA = computer architecture + communication architecture.** The communication architecture defines the
  communication and synchronisation operations (the abstraction) and their implementation.
- **Layers:** parallel applications (CAD, databases, scientific modelling) → **programming models**
  (multiprogramming, shared address space, message passing, data parallel) → communication abstraction
  (compilers/libraries) → OS → communication hardware → physical medium.
- **Generic parallel machine:** processing elements, each with memory and a **communication assist (CA)**,
  connected by an interconnection network. For shared memory the CA is tightly integrated with the memory system;
  for message passing it initiates and receives messages; for data-parallel machines it supports global
  synchronisation.

**Amdahl versus Gustafson** (derived on the slides):
```
Amdahl (fixed problem):      Speedup = 1 / ((1 − f) + f/m)                      → ≤ 1/(1 − f)
Gustafson (scaled problem):  w' = ((1 − f) + f·m)·w  ⇒  Speedup = (1 − f) + f·m       (linear in m)
```
Amdahl asks how much faster the same job gets. Gustafson asks how much more work fits in the same time.
They are both right; they answer different questions.

**The parallelisation process** (Culler/Singh/Gupta, Ch. 2):
1. **Decomposition:** split the computation into tasks. This bounds the available concurrency. Expose enough
   concurrency without excessive task-management overhead.
2. **Assignment:** distribute tasks to processes (static or dynamic). Goals: load balance, less communication,
   low runtime overhead.
3. **Orchestration:** naming, data structures, scheduling, communication, synchronisation. Goals: reduce
   communication and synchronisation cost, preserve locality, schedule to satisfy dependences early, cut
   parallelism overhead. The architecture, programming model and language matter most here.
4. **Mapping:** processes to processors (pinned or OS-managed). Goals: co-locate related processes and exploit
   network locality.

```
Speedup(p) ≤ Time(sequential work) / max over processors of (work + sync wait + comm cost + extra work)
```

**Case study — the Gauss–Seidel equation solver kernel (GSES).** On an (n+2)×(n+2) grid, sweep in place:
`A[i,j] = 0.2·(A[i,j] + A[i,j−1] + A[i−1,j] + A[i,j+1] + A[i+1,j])`, accumulate `diff += |A[i,j] − temp|`, and
repeat until `diff/(n·n) < TOL`.
- **Dependences:** each point needs the *new* values of its north and west neighbours. Within a sweep only the
  points on an **anti-diagonal** are independent, so there is O(n) concurrency and an O(n) sequential chain out of
  O(n²) work.
- **Red-black ordering:** Gauss–Seidel is already an iterative approximation, so the update order can change.
  Colour the grid like a chessboard, update all n²/2 red points in parallel, synchronise globally, then update
  the black points. The result is independent of the number of processors.
- **Decomposition:** `for_all` over both loops gives degree of concurrency n².
- **Assignment:** row-based blocks (each process gets n/p contiguous rows) or cyclic rows (process i gets rows
  i, i+p, …), or dynamic (take the next available row).
  For block rows, communication/computation = O(n)/O(n²/p) = **O(p/n)**. Square blocks give O(√p/n).
- **Orchestration, three ways:**
  - *Data parallel:* `DECOMP A[BLOCK,*,nprocs]`, `for_all`, `REDUCE(mydiff, diff, ADD)`.
  - *Shared address space:* `G_MALLOC` a shared A, `CREATE(nprocs−1, Solve, A)`, each process computes its own row
    range (`mymin = 1 + pid·n/nprocs`). Use `LOCK(diff_lock); diff += mydiff; UNLOCK`, then `BARRIER(bar1)`
    before testing convergence, a barrier after, and `WAIT_FOR_END`.
  - *Message passing:* each process allocates only its `n/nprocs + 2` rows, including **ghost rows**. Before each
    sweep it `SEND`s its boundary rows to its neighbours and `RECEIVE`s theirs. Diffs are sent to process 0, which
    decides `done` and sends it back.
- Further reading: Amdahl (AFIPS 1967); Gustafson (CACM 1988); Hill & Marty (Computer 2008, see L16); Sun & Chen
  (JPDC 2010).

## L4 (31 Jan) — Parallel memory system design

**Inclusion policies** (single core): **inclusive** (L1 ⊆ LLC; an LLC eviction must back-invalidate L1),
**exclusive** (L1 ∩ LLC = ∅; L1 victims go to the LLC, which acts as a victim cache), **non-inclusive** (no
guarantee). AMAT = HT_L1 + MR_L1 × (HT_LLC + MR_LLC × MP_LLC).

**Multicore organisations:** private L1s with a shared LLC (centralised shared memory, UMA, bus-based) versus
**distributed shared memory** (each node has memory; NUMA; interconnection network).
Multicore caches reduce average access time and bandwidth demand, and let several processors load and store
concurrently.

**Managing a large LLC: shared or private?**

| | Shared LLC | Private LLC |
|---|---|---|
| Capacity use | dynamic allocation among cores; no replication | **replication** of shared blocks wastes capacity |
| Interference | **core-to-core interference** (one thread can evict another's data) | none |
| Contention | high contention at the shared resource | none |
| Hit latency | high (long wires, banking) | low |

- **Shared-private hybrids:** cluster cores, e.g. 4 cores share one LLC slice with snooping between clusters.
- **Centralised shared LLC:** multi-banked, with the controller replicated per bank (e.g. 8 cores around 64 banks).
- **Distributed shared LLC (tiled):** each tile has a core, L1s and an LLC slice, connected by a NoC. This scales
  and cuts verification cost, because one tile is replicated.
- **UCA vs NUCA:** a uniform cache access design must assume worst-case delay (H-tree banking). A
  **non-uniform** design (NUCA; Kim, Burger & Keckler, ASPLOS 2002) connects banks with a NoC, so near banks are
  faster. Data management raises three issues: **mapping** (where a block may live), **searching** (how to find
  it) and **movement** (migrating blocks towards their users).
  **S-NUCA** maps statically by address bits. **D-NUCA** lets blocks migrate towards the requester, so
  efficient search becomes the open problem.
- **The three problems shared memory raises:** cache coherence (L5), synchronisation (L7) and memory consistency
  (L6).

## L5 (5 Feb) — Cache coherence

**The problem:** private caches can hold different values for the same address. The example has u = 5 in memory;
cores 1 and 3 read it, core 3 writes 7, then cores 1 and 2 read. With write-through caches core 1 reads a stale
copy. With write-back caches both core 1 and core 2 can read stale values.

**Coherence invariants** (Primer): **SWMR** — at any instant, for any location, either one core may write (and
read) or any number may only read. **Data-value invariant** — the value at the start of an epoch equals the value
at the end of the last read-write epoch. Classical formulation: **write propagation** plus **write
serialisation**.

**Two families:** **write-invalidate** (gain exclusive access before writing; the dominant approach) and
**write-update** (broadcast new values to all copies). The table on the slide pairs bus-based shared memory or
distributed shared memory × write-back or write-through caches × invalidation or update protocols.
Implementation: **snooping** (a broadcast medium; every controller watches every transaction) or a
**directory** (point-to-point; the home node tracks sharers).

**Snoopy implementation:** each cache line has coherence-state bits. The controller updates them in response to
**processor events** (PrRd, PrWr) and **snooped bus events** (BusRd, BusRdX, BusUpgr, Flush), and issues bus
transactions.

### MSI (write-back, invalidation)

| State / event | PrRd | PrWr | BusRd (snooped) | BusRdX (snooped) |
|---|---|---|---|---|
| **I** | → S, issue BusRd | → M, issue BusRdX | – | – |
| **S** | hit | → M, issue BusRdX (or BusUpgr) | stay S | → I |
| **M** | hit | hit | → S, **Flush** | → I, **Flush** |

Replacing an M block writes it back (BusWB / Flush). Replacing an S block is silent.

### MESI (adds **E**xclusive-clean)

- **E** means this is the only cached copy and it is clean, so a write can proceed **silently** (E→M with no bus
  transaction). This saves the upgrade for private data, the common case.
- How does the requester know nobody else has the block? A **shared line** (wired-OR). During the BusRd address
  phase every cache holding a copy asserts it. The requester loads the block in **E** if the line is not asserted
  and in **S** if it is.
- Transitions to know: I→E on `PrRd/BusRd(¬S)`; I→S on `PrRd/BusRd(S)`; E→M on PrWr (silent); E→S on snooped
  BusRd; E→I on snooped BusRdX.
  **E is clean, so there is no data to flush.** The diagram's "Flush′" only applies when cache-to-cache transfer
  is used. (This exact point is the bug in my Assignment 2 submission; see Part E.)

### MESIF (Intel)

Adds **F**orward: among the sharers, exactly one holds the block in F and answers read requests (cache-to-cache),
so memory and the other sharers stay quiet. The **newest** requester gets F and the old forwarder drops to S, so
F is unlikely to be evicted soon and forwarding bandwidth for hot blocks spreads across cores.

### MOESI (AMD)

Adds **O**wned. A dirty block can be **shared without writing it back to memory first**. The owner (O) supplies
data and is responsible for the eventual write-back; the other copies are S, and the memory copy may be stale.

### Dragon (write-back *update*; Xerox Dragon, Sun SparcServer)

States **E** (exclusive clean), **Sc** (shared clean), **Sm** (shared modified: this cache owns the write-back
responsibility) and **M** (exclusive modified). There is **no invalid state**: copies are updated with
**BusUpd**, never invalidated. Update protocols suit fine-grained producer-consumer sharing, but waste bandwidth
when one core writes a block repeatedly that others no longer read.

### Directory-based coherence (scalable)

- Each memory block has a directory entry: **presence bits p[1..n] plus a dirty bit** (the full bit-vector
  scheme). On a miss, the core contacts the home directory, which finds the valid copies, decides further actions
  (invalidations, forwarding) and updates itself.
- Directory states: **Uncached**, **Shared** (read-only, sharers set) and **Modified/Exclusive** (one owner).
  - Uncached + read miss → reply with data; Sharers = {P}; → Shared.
  - Uncached + write miss → reply with data; Sharers = {P}; → Modified.
  - Shared + read miss → reply with data; Sharers += {P}.
  - Shared + write miss → **invalidate** all sharers, reply with data; Sharers = {P}; → Modified.
  - Modified + read miss → **fetch** from the owner (who writes back and goes to S), reply; Sharers += {P};
    → Shared.
  - Modified + write miss → **fetch/invalidate** the owner, reply; Sharers = {P}.
  - Modified + data write-back → Sharers = {}; → Uncached.
- **Storage overhead:** with 64 B blocks and 64 cores, the presence vector is 64 bits per 512 bits of data, a
  12.5% overhead that grows linearly with core count. Remedies: limited-pointer directories, coarse vectors,
  sparse directories (directory caches) and snoop filters.
- Transactions take **3 hops** (requester → home → owner → requester, with forwarding) or **4 hops**. NoC latency
  and directory bandwidth are the scaling limits (see L9).

## L6 (7 Feb) — Memory consistency

**Why coherence is not enough:** coherence is per-block. Consistency specifies the allowed orderings of loads and
stores **across different addresses**, which programmers need to reason about correctness. It also tells the
system designer how much the compiler and hardware may reorder.

**Two litmus tests to know by heart:**

| Test | Core C1 | Core C2 | Forbidden under SC |
|---|---|---|---|
| **MP (message passing)** | `S1: data = NEW; S2: flag = SET` | `L1: r1 = flag; if (r1 != SET) goto L1; L2: r2 = data` | (r1, r2) = (SET, 0) |
| **SB (store buffering / Dekker)** | `S1: x = NEW; L1: r1 = y` | `S2: y = NEW; L2: r2 = x` | (r1, r2) = (0, 0) |

**Sequential consistency** (Lamport 1979): *the result of any execution is the same as if the operations of all
processors were executed in some sequential order, and the operations of each processor appear in this sequence
in program order.* All four orders are preserved: Load→Load, Load→Store, Store→Store, Store→Load. The
implementation model is cores issuing in program order to a "switch" that picks one core at a time.

**Sufficient conditions for SC** (Adve & Gharachorloo):
1. Every process issues memory operations in program order.
2. **Write completion:** after a write is issued, the process waits for it to complete before its next operation.
3. **Write atomicity:** after a read, the process waits both for the read to complete and for the write whose
   value it returned to complete (be visible to all), before issuing its next operation.
   Example: P1 `A = 1`; P2 `while (A == 0); B = 1`; P3 `while (B == 0); print A` must print 1.

**A snoopy MSI system provides SC:** a write completes when its BusRdX or invalidation appears on the bus. Write
atomicity holds because the bus totally orders transactions, and a read that returns a written value comes after
that write's bus transaction.

**Write buffers break SC:** each core keeps committed stores in a FIFO write (store) buffer, and its loads bypass
it. In the SB test both loads can read 0 before either buffered store drains, giving (0, 0), which SC forbids.
This is harmless on a uniprocessor but visible on a multicore.

**TSO (Total Store Order; x86, SPARC TSO):** preserves Load→Load, Load→Store and Store→Store, but **relaxes
Store→Load**. Implementation: a per-core FIFO store buffer with load bypassing (a core sees its own buffered
stores). On a multithreaded core, one thread context must never bypass from **another** context's store buffer,
because that would expose stores early and break TSO's total store order.
- MP is still safe under TSO (stores leave in order and loads stay in order); SB can show (0, 0).
- **Atomic RMWs under TSO:** an RMW is a load immediately followed by a store. Its load must not pass earlier
  loads, nor earlier stores, so the RMW **drains the store buffer** before its load. The load needs read-write
  coherence permission, and the controller must not give up the block between the load and the store. On x86
  this is why `lock`-prefixed instructions act as full fences.
- To restore SC ordering for SB under TSO, put an **MFENCE** (or a locked RMW) between the store and the load.

**Relaxed models:** the Primer's example model **XC** has a `FENCE` instruction and respects program order only
around fences (Load→FENCE, Store→FENCE, FENCE→FENCE, FENCE→Load, FENCE→Store). Same-address accesses follow TSO
rules: loads see the core's own earlier stores. The implementation is a reorder unit between each core and the
coherent memory system.
- In MP, programmers only need S1, S2 → S3 → L1 and L1 → L2, L3. SC and TSO also force S1→S2 and L2→L3, which
  are unnecessary. With `F1: FENCE` between the data stores and the flag store, and `F2: FENCE` after the spin
  loop, XC gives S1, S2 → F1 → S3 → L1 loads SET → F2 → L2, L3.
- Real relaxed models: ARMv8 (other-multi-copy-atomic, with `DMB`/`DSB`, load-acquire and store-release),
  RISC-V RVWMO (`fence rw,rw`, `.aq/.rl` bits), Power (non-multi-copy-atomic, `sync`/`lwsync`).
  Language models such as C++11 (`memory_order_seq_cst/acq_rel/acquire/release/relaxed`) are mapped onto these by
  compilers. **Data-race-free (DRF) programs get SC semantics** on all these models (SC-for-DRF).

## L7 (12 Feb) — Synchronisation and transactional memory

**Types:** mutual exclusion (lock/unlock), point-to-point events (flags), global events (barriers).
Hardware gives speed; software gives cost, flexibility and adaptability.

- **A broken software lock:** `ld r,loc; cmp r,#0; bnz lock; st loc,#1`. Two processes can both read 0 before
  either stores 1, so **both enter**. The read-test-write must be atomic.
- **Atomic RMW primitives:** test-and-set, swap, fetch-and-increment, compare-and-swap.
  `lock: t&s reg, loc; bnz reg, lock; ret` and `unlock: st loc, #0`.
  - Plain T&S spinning causes a coherence storm: every T&S is a BusRdX. Use **test-and-test-and-set** (spin on a
    cached read, then T&S only when the lock looks free), exponential back-off, or queue locks (ticket lock,
    MCS/CLH) for scalability and fairness.
- **ISA support:** IBM 370 compare&swap; x86 `lock` prefix on many instructions (`lock cmpxchg`, `xadd`);
  SPARC swap and CAS; MIPS, Power, ARM and RISC-V use **LL/SC** (load-linked/load-reserved,
  store-conditional).
- **LL/SC semantics:** LL reads a location and sets a reservation. Arbitrary instructions follow. SC writes only
  if no other write to that location has happened since the LL, and it returns success or failure. If SC
  succeeds, the sequence was atomic. If it fails, it doesn't write and generates no invalidations.
  Lock: `lock: ll r1,loc; bnz r1,lock; sc loc,r2; beqz lock; ret` (with r2 = 1).
- **LL/SC implementation:** a per-processor **lock-flag** plus **lock-address register**. LL sets the flag and
  records the block address. Incoming invalidations or updates that match the address clear the flag, and so do
  eviction of the block and a context switch between LL and SC. SC checks the flag. LL/SC avoids the ABA problem
  (L8) because *any* intervening write breaks the reservation.
- **Transactional memory:** a transaction is a sequence of reads and writes that appears **atomic** (all or
  nothing) and **isolated** (no one sees intermediate states). This avoids lock overhead, poor scalability,
  deadlock and starvation.
  - **HTM:** the hardware tracks read and write sets (e.g. via cache tag bits) and aborts and rolls back on a
    conflict. It is simple and fast, and free from deadlock and livelock, but limited in capacity (set size,
    cache associativity) and in conflict resolution. Examples: Intel TSX/RTM (now largely disabled after
    errata and side channels), IBM POWER8/zEC12, ARM TME.
  - **STM:** the runtime versions objects and validates read sets. It is portable and supports fine-grained
    parallelism, but has overhead and complexity.

## L8 (19 Feb) — Hardware support for programming and debugging

- **Hardware multithreading:** a thread context is a PC plus registers. Resources fall into three groups:

| Category | Resources |
|---|---|
| Replicated | PC, architectural registers, register-renaming logic |
| Partitioned | ROB, load/store buffers, queues (fetch, issue) |
| Shared | caches, physical registers, execution units |

  Key characteristics are the thread-scheduling policy and the pipeline partitioning. Implementations:
  coarse-grain (switch on long stalls), fine-grain (interleave every cycle) and **SMT** (issue from several
  threads in the same cycle).
- **Lock-free programming with atomics:** CAS lets threads update shared structures without locks, giving progress
  guarantees, no deadlock and less context switching. **Lock-freedom** means at any time at least one thread can
  make progress. Examples: lock-free stack, queue (Michael–Scott), hash table.
  - **The ABA problem:** a thread reads A and prepares `CAS(A → C)`. Meanwhile others change A→B→A, and the CAS
    wrongly succeeds (the slide shows CAS(3,8), CAS(8,3), then a stale CAS(3,6)). Fixes: **LL/SC** (it detects any
    intervening write), tagged/versioned pointers (double-width CAS), and **hazard pointers** or epoch-based
    reclamation for safe memory reclamation.
- **Memory barriers:**

| Barrier | x86 | ARM | Prevents |
|---|---|---|---|
| Full | `MFENCE` | `DMB SY` | all reordering |
| Store-store | `SFENCE` | `DMB ST` | store-store reordering |
| Load-load | `LFENCE` | `DMB LD` | load-load reordering |

  Nuance: x86 is TSO, so for ordinary write-back memory only Store→Load reordering can happen and `MFENCE` (or a
  locked instruction) is the one that matters. `SFENCE` orders non-temporal or write-combining stores, and
  `LFENCE` is mostly used as an execution or speculation barrier (Spectre mitigations).
- **Debugging support — performance monitoring counters (PMCs):** count instructions, cache hits and misses,
  branch mispredictions, stalls. Intel exposes them through MSRs (perf, VTune, PCM); AMD adds IBS
  (instruction-based sampling: perf, uProf); ARM through `PMCR_EL0` (perf, Streamline). Uses: tuning, finding
  memory bottlenecks, attack detection, power management.
- **Trace buffers:** low-overhead execution tracing. Intel PT (memory-mapped buffers via ToPA, the Table of
  Physical Addresses), AMD (LBR, last-branch record), ARM **CoreSight** (ETB/TMC/ETR trace RAM).

## L9 (21 Feb) — Manycore processors

Examples (2025):

| Chip | Cores / threads | Clock | Cache | Memory | TDP | ISA |
|---|---|---|---|---|---|---|
| AMD EPYC 9965 | 192 / 384 | 2.25–3.7 GHz | 384 MB L3 | 12-ch DDR5 | 500 W | x86 |
| AmpereOne A192-32X | 192 / 192 | 3.2 GHz | 64 MB SLC* | 8-ch DDR5 | 400 W | ARMv8.6+ |
| Intel Xeon 6980P | 128 / 256 | 2.0–3.9 GHz | 504 MB L3 | 12-ch DDR5 | 500 W | x86 |
| Intel Xeon 6780E | 144 / 144 (E-cores) | 2.2–3.0 GHz | 108 MB L3 | 8-ch DDR5 | 330 W | x86 |

(*The slide lists "L2 Cache 384 MB" for AmpereOne, which is the 192 × 2 MB per-core L2 total.)

**Challenges:**
1. **Shared-resource contention:** memory-hierarchy partitioning versus utility-based allocation (e.g. Intel CAT
   way-partitioning; Qureshi & Patt's UCP), and the OS's role in co-scheduling. (Surveys: Zhuravlev et al. CSUR
   2012; Kundan et al. TACO 2022; Divide&Content, TPDS 2023.)
2. **Bandwidth wall:** off-chip bandwidth grows much more slowly than core counts (pins and power). Responses are
   bandwidth-conservation techniques (bigger and smarter caches, compression, DRAM caches) and bandwidth-aware
   core allocation (Rogers et al., ISCA 2009; Wang et al., HPCA 2016).
3. **Coherence at scale:** directories carry bandwidth, latency and storage costs. Research includes Tardis
   (timestamp-based coherence, PACT 2015) and optimised home-agent mappings (IEEE Access 2021).

## L10 (26 Feb) — How cores are connected: interconnection networks (NoCs)

**Real chips:**
- **AMD EPYC 9965:** 12 CCDs (core-complex dies, Zen 5c, each a single CCX with a shared L3) around a central
  **I/O die** with memory controllers and Infinity Fabric (GMI) links. It is chiplet-based and NUMA-partitionable
  (NPS modes).
- **AmpereOne:** a monolithic mesh of core clusters with distributed SLC slices.
- **Intel Xeon 6980P:** three compute chiplets plus I/O chiplets. Each core tile has a 2 MB L2 and a CHA/LLC slice
  (coherence + 4 MB LLC) on a **mesh** fabric (a router per tile).

**Why NoCs:** wire delay doesn't scale (global wires get relatively slower than gates, as the chart of interconnect
length and delay by technology node shows). Buses have limited bandwidth, high power to drive long wires, and a
centralised arbiter. Crossbars fix bandwidth but scale as O(n²) in area and power. **NoCs** give scalable
bandwidth at low area and power, use wires efficiently, and have regular topologies with short local links
(Intel 80-core Teraflops chip; Tilera).

**Building blocks:** topology, routing, flow control, router microarchitecture.

### Topology

Topology sets the hop count, link lengths, path diversity and cost (node degree, wiring).

| Topology (k×k = N nodes) | Degree | Diameter | Bisection (links) |
|---|---|---|---|
| Ring (N) | 2 | ⌊N/2⌋ | 2 |
| 2D mesh | ≤ 4 | 2(k−1) | k |
| 2D torus | 4 | 2⌊k/2⌋ | 2k |
| Folded torus | same as torus, with equal-length links (no long wrap-around wire) | | |

### Routing

- **Deterministic or oblivious** versus **adaptive**; **minimal** versus **non-minimal**.
- **Dimension-ordered routing (DOR):** X-Y routes fully in X, then in Y. It is simple and **deadlock-free**,
  because it forbids all Y→X turns, so no cyclic channel dependency can form. Its weaknesses are low throughput
  and load imbalance under adversarial traffic.
- **Turn model** (Glass & Ni): prohibit just enough turns to break every cycle. **West-first** routes all
  westward hops first, then adapts. Adaptive routing can use local or global congestion information, but may
  reorder messages and must avoid deadlock (cyclic resource dependencies) and livelock (endless non-minimal
  wandering).
- **Implementation:** **source routing** puts the whole route (e.g. "EENX") in the header. Per-hop delay is lower,
  but every network interface needs a table. **Node (table) routing** keeps a small table per router (e.g. a
  3×3 mesh with west-first entries). It is reconfigurable and can react to congestion, but adds per-hop delay.

### Flow control

Granularity: **message → packets** (header with route and sequence number, plus payload) **→ flits** (head,
body, tail, or a single head-tail flit; type and VC-ID fields) **→ phits** (what a link carries per cycle).

| Technique | Allocation unit | Zero-load latency (H hops, L bits, bandwidth b, router delay t_r) | Notes |
|---|---|---|---|
| Store-and-forward | packet buffers | H × (t_r + L/b) | waits for the whole packet at every hop |
| Virtual cut-through | packet buffers | H × t_r + L/b | forwards the head as soon as possible; needs whole-packet buffers to absorb a blocked packet |
| Wormhole | flit buffers | H × t_r + L/b | tiny buffers, but a blocked packet occupies several routers (head-of-line blocking) |

**Virtual channels** (Dally 1992): several flit queues share one physical channel. A VC is allocated to the head
flit at each router and the body and tail flits inherit it, so flits of different packets interleave. VCs fix
head-of-line blocking, work with any flow-control scheme, and **break deadlock**, e.g. through escape VCs or
dateline VCs in tori.

**Credit-based VC router:** input buffers per VC, route computation (RC), VC allocator (VA), switch allocator (SA),
crossbar (ST), link traversal (LT). Credits flow back upstream to tell the sender how many free buffer slots
exist. The classic **5-stage pipeline:** the head flit goes BW (buffer write) → RC → VA → SA → ST → LT; body and
tail flits skip RC and VA. Speculation and look-ahead routing shorten it to 2–3 stages.

## L11 (5 Mar) — Dark silicon

My handwritten notes on the slides: *Dennard scaling — power density stays constant as transistors shrink, giving
faster and more energy-efficient chips each generation. Consequences: power stays constant, performance increases,
utilisation remains 100%. "Utilisation wall": even though Moore's law gives more transistors, power constraints
force a large fraction of them to remain dark. Dark = idle (off); dim = under-clocked.*

**Scaling algebra** with S = S_node / S_scaled-node (about 1.4 per generation):

| Property | Dennard | Post-Dennard |
|---|---|---|
| Transistor count | S² | S² |
| Frequency | S | S |
| Capacitance | 1/S | 1/S |
| V_dd² | 1/S² | **1** (V_dd stuck: leakage and reliability limits) |
| Power (= Q·F·C·V²) | 1 | **S²** |
| Utilisation at fixed power | 1 | **1/S²** |

So the fraction of a chip that can switch at full frequency drops exponentially with each generation (Venkatesh
et al., "Conservation Cores", ASPLOS 2010; Taylor, "Is dark silicon useful?", DAC 2012). Going from 65 nm to 32 nm
(S = 2), you either get 4 cores at 1.8 GHz with 12 cores dark (the industry's "dark" choice), or 2×4 cores at
0.9 GHz (dim). The dark fraction is about 75% after 2 generations and about 93% after 4.
**Dark silicon doesn't mean useless silicon. It is silicon that can't all be used all the time at full frequency.**

**Taylor's "four horsemen" of the dark-silicon apocalypse** (my note: shrinking silicon, dim silicon, specialised
cores, new circuit paradigms):
1. **Shrinking chips:** build smaller dies instead of dark area (silicon is expensive). Downside: exponentially
   rising power density.
2. **Dim silicon:**
   - *Spatial dimming:* more cores at lower frequency, or near-threshold voltage (NTV) operation. My note: NTV
     only pays off for highly parallel workloads.
   - *Temporal dimming:* **computational sprinting** (Raghavan et al., HPCA 2012) and Turbo Boost. Run hot
     briefly using thermal capacitance, then idle. ARM big.LITTLE is temporal and spatial heterogeneity at once.
3. **Specialised cores/accelerators:** spend dark area on **domain-specific units that run only when needed**.
   Examples are conservation cores and GreenDroid (Goulding-Hotta et al., IEEE Micro 2011). My note: by replacing
   many small hardware blocks and turning them on only for their data, a chip "fills" its dark regions with useful
   engines, raising throughput or lowering system power.
4. **New circuit paradigms:** neuromorphic or spiking circuits, analog compute-in-memory (e.g. NeuRRAM),
   approximate circuits.

The "greenest computer" contrast: Frontier (≈21 MW, 1.102 EFLOP/s) versus the human brain (about 20 W at rest).
Going forward: heterogeneous architectures (DSAs, CPU/FPGA, big.LITTLE, CPU/GPU) and emerging concepts (PIM,
neuromorphic, approximate computing, novel memories, 3D stacking, chiplets, graph-centric). My note: modern SoCs
combine big and LITTLE cores, GPUs, DSPs and fixed-function units on one die. GPUs exploit data parallelism, DSPs
handle signal pipelines, FPGAs reconfigure for tasks, and TPUs and Apple's Neural Engine are ASICs for neural nets.

---

# Part C — Heterogeneous computer architecture

## L12 (12 Mar) — Domain-specific architectures (H&P Ch. 7) and the TPU

**What Moore's law bought:** cache hierarchies, wide SIMD units, deeper pipelines, branch prediction, OoO
execution, speculative prefetching, multithreading, multiprocessing. Moore (2005): *"Moore's Law can't continue
forever … another 10 or 20 years before we reach a fundamental limit."*

**Consequences of the end of Dennard scaling** (my note: constant power density is gone): more switching
transistors means more power, so one inefficient processor is replaced by many efficient cores. **Energy per
operation** shows where the waste is (Dally/H&P figure):

| Operation | Energy |
|---|---|
| RISC instruction (mostly overhead: fetch, decode, control, register access) | ~125 pJ |
| Load/store (D-cache) | ~150 pJ |
| SP floating point | 15–20 pJ |
| 32-bit addition | 7 pJ |
| 8-bit addition | 0.2–0.5 pJ |

The ALU work is a tiny fraction of an instruction's energy, so we must **raise the number of arithmetic
operations per instruction from one to hundreds**. General-purpose cores give way to **domain-specific
architectures (DSAs)**. My note: end of Dennard scaling → limits of general-purpose scaling → need for DSAs, with
AI/ML workloads driving TPUs and neural accelerators.

Context: "data is the new oil". AI compute demand doubled roughly every 3.4–6 months in the deep-learning era,
versus Moore's 2 years, and transformer parameter counts went from millions to trillions.
**The AI kernel is the fully connected layer: Y = activation(W·X + B)**, i.e. matrix-vector or matrix-matrix
multiply. Hardware evolved from scalar (CPU) to vector (GPU SIMT) to **tensor** (TPU, Tensor Cores) units, each
doing more MACs per instruction.

**Five guidelines for DSAs** (H&P):
1. **Dedicated memories** to minimise data-movement distance: hardware-controlled multi-level caches become
   software-controlled **scratchpads** (my note: GPU shared memory is the same idea).
2. **Spend resources on more arithmetic units or bigger memories**, not on OoO, speculation or multithreading.
3. **Use the easiest form of parallelism** that matches the domain: SIMD or VLIW instead of MIMD.
4. **Reduce data size and type** to the simplest the domain needs: 8- or 16-bit integer or float instead of 32/64.
5. **Use a domain-specific programming language**: TensorFlow, Halide, PyTorch instead of C/C++/Python loops.

| Guideline | TPU v1 | Pixel Visual Core | Microsoft Catapult |
|---|---|---|---|
| Design target | datacenter ASIC | mobile SoC IP | datacenter FPGA |
| Dedicated memories | 24 MB Unified Buffer, 4 MB accumulators | 128 KB line buffer per core | varies |
| Larger arithmetic | **65,536 MACs** | 256 MACs per core | varies |
| Easy parallelism | single-threaded, SIMD, in-order | MPMD, SIMD, VLIW | SIMD, MISD |
| Smaller data | 8/16-bit int | 8/16/32-bit int | 8/16-bit int, 32-bit FP |
| DSL | TensorFlow | Halide/TensorFlow | Verilog |

### TPU v1 (Jouppi et al., ISCA 2017)

- Google's first custom DSA for warehouse-scale computers (deployed in 2015, built for **inference**). It is a
  **coprocessor on PCIe**: the host sends instructions over PCIe into an instruction buffer, so the TPU has no
  fetch of its own.
- **Matrix Multiply Unit:** a **256×256 systolic array of 8-bit MACs** (65,536), giving 92 TOPS peak at 700 MHz
  in 28 nm at about 40 W. Signed or unsigned 8-bit inputs; 16-bit products accumulate into 32-bit accumulators.
- **Memories:** a 24 MiB on-chip **Unified Buffer** for activations, 4 MiB of **accumulators**, and a **Weight
  FIFO** fed from 8 GiB of off-chip DDR3 **Weight Memory** (about 30 GB/s, the bottleneck for MLPs and LSTMs,
  which run memory-bound under the roofline). Internal paths are 256 B wide. A programmable DMA controller
  completes it.
- **ISA:** CISC with CPI about 10–20, **no PC and no branches** (straight-line execution; my note: one instruction
  follows the next). Five key instructions out of about a dozen:
  `Read_Host_Memory` (CPU memory → Unified Buffer), `Read_Weights` (Weight Memory → Weight FIFO),
  `MatrixMultiply/Convolve` (a B×256 input times a 256×256 constant weight tile gives a B×256 output in B pipelined
  cycles, into the accumulators), `Activate` (accumulators → nonlinearity, normalise/pool → Unified Buffer) and
  `Write_Host_Memory`. The pipeline overlaps these four phases.
- **Systolic array** (H. T. Kung, "Why systolic architectures?", 1982): **weight-stationary**. Weights are
  preloaded into the MAC cells; activations flow in from the left, staggered diagonally; partial sums flow down
  and leave at the bottom into the accumulators. Each operand is read from SRAM once and reused about 256 times
  as it moves cell to cell, cutting register-file and SRAM energy. The animation on the slide shows
  y₁ = w₁₁x₁ + w₁₂x₂ + w₁₃x₃ forming as data marches through.
- Result: 15–30× faster and 30–80× better performance per watt than the contemporary K80 GPU and Haswell CPU on
  Google's inference workloads. Later TPUs (v2–v7) added training, HBM, bf16 (invented for TPUs), inter-chip
  interconnect (ICI) tori, and optical circuit switches.

## L13–14 (26–28 Mar) — Data-level parallelism and GPUs

**Flynn's taxonomy:**
- **SISD:** uniprocessors (ILP).
- **SIMD:** DLP (vector machines, multimedia extensions, GPUs).
- **MISD:** rare; the slide cites Microsoft Catapult's pipelined feature-extraction FSMs.
- **MIMD:** TLP. Tightly coupled multicores, plus loosely coupled request-level parallelism in warehouse-scale
  computers.

**SIMD:** exploits DLP in matrix and media code. It is **more energy-efficient than MIMD** (one instruction
fetch/decode drives many data lanes), and programmers can still think sequentially. Variants are vector
architectures, multimedia SIMD extensions and GPUs.

**Multimedia SIMD extensions:** narrow data types lead to "partitioned" adders that break carry chains. Intel
**MMX** (1996): eight 8-bit or four 16-bit operations in 64-bit registers. **SSE** (1999): separate 128-bit XMM
registers, 16×8, 8×16 or 4×32 integer ops, 4×32 or 2×64 FP ops, separate data-transfer instructions. **AVX**
(2010): 256-bit registers, 4×64-bit FP. Later AVX-512 and AMX (tile matrix units).
`daxpy` (`y[i] = a*x[i] + y[i]`) is the canonical example.

**Heterogeneous CPU+GPU model:** the sequential part runs on the CPU (big caches, few cores); the
computation-intensive part is offloaded to the GPU (thousands of simple lanes, small caches, high bandwidth).

**GPUs (2025):** GeForce RTX 5090 has 21,760 CUDA cores, 32 GB GDDR7 on a 512-bit bus, 1,792 GB/s. Radeon RX 9070
XT has 64 CUs and 4,096 SPs, 16 GB GDDR6, 640 GB/s. **Discrete versus integrated:** the old northbridge/PCIe
picture versus unified-memory SoCs (Apple M4 Max: 12P+4E CPU cores, a 40-core GPU and a Neural Engine sharing one
memory pool).

**NVIDIA Hopper GH100:** the full die has **8 GPCs, 72 TPCs and 144 SMs** (the H100 SXM product enables 132),
with a 50 MB L2 and HBM3. **Hopper SM:** four partitions, each with a warp scheduler, a 16K×32-bit register file,
32 FP32 and 16 INT32/FP64 units, and a **4th-gen Tensor Core**. The SM has 256 KB of combined L1/shared memory
and a **Tensor Memory Accelerator** (TMA, asynchronous bulk copies).
Tensor-core types: INT8, **FP8 (E4M3 and E5M2)**, FP16, BF16, TF32, FP64. Know the bit layouts:

| Format | Sign | Exponent | Mantissa |
|---|---|---|---|
| FP32 | 1 | 8 | 23 |
| FP16 | 1 | 5 | 10 |
| BF16 | 1 | 8 | 7 (FP32's range, low precision) |
| FP8 E5M2 | 1 | 5 | 2 |
| FP8 E4M3 | 1 | 4 | 3 |

**Programming GPUs:** a heterogeneous model in which the CPU is the **host** and the GPU the **device**. CUDA (or
vendor-neutral OpenCL) unifies multithreading, MIMD, SIMD and ILP as **CUDA threads**. The execution model is
**SIMT** (single instruction, multiple threads).
- One **thread** per data element, with no need for synchronisation between them.
- Threads group into **thread blocks** (CTAs), executed by a multithreaded SIMD processor (an SM). Blocks run
  independently and in any order. Each block is split into **warps** of 32 threads. Threads in a block
  communicate through **shared memory** and `__syncthreads`.
- Blocks form a **grid**; the programmer chooses grid and block dimensions.
- **Thread block clusters** (Hopper): blocks guaranteed to be co-scheduled on SMs within one GPC. They can access
  each other's shared memory over an SM-to-SM network (**distributed shared memory**) instead of going through
  global memory as on A100.
- **Mapping example:** a CUDA program with 4,096 blocks of 256 threads; each block has 8 warps; the warp
  schedulers pick ready warps each cycle.

**GPU memory hierarchy:**
- **Register file:** per thread, the fastest level; register pressure limits occupancy.
- **Shared memory + L1:** on-chip and per SM, with configurable split.
- **L2:** shared by all SMs; caches global memory.
- **Constant cache:** read-only and broadcast-friendly.
- **Texture cache:** for non-uniform, scattered reads (also used by AI workloads).
- **Global memory:** HBM3 or GDDR. Accesses should be **coalesced** across a warp, and can be cached in L1/L2.

Memory scopes: per-thread local memory, per-block shared memory, per-cluster distributed shared memory, and global
memory shared across grids (with inter-grid synchronisation). See `CS6023_GPU_Learning.md` for CUDA in depth.

## L15 (28 Mar) — FPGAs and CPU–FPGA hybrid systems

- **Datacenter pressures:** CPU performance bottlenecks, energy inefficiency (datacenters use about 1–1.5% of
  world electricity and are growing fast with AI), and limited scalability and flexibility. Accelerators sit on a
  spectrum from flexibility to efficiency: CPU → GPU → FPGA → ASIC (Microsoft Build 2017).
- **Vendors:** Altera (bought by Intel for $16.7B in 2015) and Xilinx (bought by AMD for about $35–49B; announced
  2020, closed 2022). Users: Azure (AI inference, Bing, networking via SmartNICs), AWS F1 (genomics, finance,
  video), Meta, Baidu.
- **FPGA fabric:** an array of **configurable logic blocks (CLBs)** in a sea of programmable interconnect (switch
  boxes), ringed by I/O blocks. A **k-input LUT** is 2^k SRAM bits feeding a 2^k:1 multiplexer, so it can
  implement *any* k-input Boolean function (the slide shows a 2-input LUT with 4 SRAM bits and a 4:1 mux).
  Modern FPGAs add DSP slices (MAC units), block RAMs, hard PCIe and Ethernet, and HBM.
- **Single-event upsets (SEUs):** an FPGA's configuration lives in SRAM, and particle strikes can flip a bit and
  thereby change the circuit. Combinational SEUs cause transient glitches, while direct hits on flip-flops cause
  soft faults. Mitigations: **technology hardening** (extra doping layers), **circuit hardening** (radiation-hard
  cells) and **system hardening** (**TMR** triple-modular redundancy with voting, configuration scrubbing,
  ECC/EDAC).
- **CPU–FPGA platforms** (Choi et al., ACM TRETS 2019):
  - (a) Conventional PCIe cards with private device DRAM and explicit DMA (Alpha Data, AWS F1).
  - (b) **PCIe-based CAPI** (IBM POWER8). The CAPP (Coherent Accelerator Processor Proxy) on the CPU keeps the
    off-chip accelerator coherent; the PSL (Power Service Layer) on the FPGA handles address translation and
    coherence.
  - (c) **QPI-based Xeon+FPGA v1** (Intel HARP), with coherent shared memory.
  - (d) Xeon+FPGA v2 with 1 QPI plus 2 PCIe links. Today **CXL** plays this role.

  Terms: AFU = accelerator functional unit; FIU = FPGA interface unit; SPL = system protocol layer (address
  translation, request reordering).
- **Microsoft Catapult** (Putnam et al., ISCA 2014; Caulfield et al., "Configurable Cloud", MICRO 2016). Catapult
  v2 places an FPGA as a **bump-in-the-wire** between the server NIC and the top-of-rack switch (40 Gb/s QSFP),
  plus PCIe to the CPU and local DDR3. RTL is split into a **shell** (reusable I/O, DMA, DDR controllers, network
  routing) and a **role** (the application logic). This eases programming and enables partial reconfiguration.
  - **Bing ranking feature extraction:** a hit-vector preprocessing FSM feeds a large array of
    **feature-extraction FSMs** (MISD style) that compute query–document features in parallel, then a
    feature-gathering network. Catapult v1 nearly doubled ranking throughput per server at fixed latency.
    Brainwave (ISCA 2018) later ran DNN inference on the same fabric.

## L16 (2 Apr) — Amdahl's law in the multicore era (Hill & Marty, IEEE Computer 2008) and big.LITTLE

**Model:** a chip has n **BCEs** (base-core equivalents). A 1-BCE core has performance 1. An "enhanced" core of
r BCEs has performance perf(r), and **Pollack's rule** gives perf(r) = √r. If perf(r) > r it helps both serial
and parallel code; if perf(r) < r, as with √r, it helps serial code but hurts parallel throughput per BCE.

```
Symmetric  (n/r cores of r BCEs):   Speedup = 1 / ( (1−f)/perf(r) + f·r/(perf(r)·n) )
Asymmetric (1 big r-BCE core + (n−r) 1-BCE cores; serial runs on big, parallel uses all):
                                    Speedup = 1 / ( (1−f)/perf(r) + f/(perf(r) + n − r) )
Dynamic    (fuse r BCEs for serial, all n for parallel; from the paper):
                                    Speedup = 1 / ( (1−f)/perf(r) + f/n )
```

Takeaways (from the n = 16 and n = 256 curves):
- **Finding parallelism (f → 1) is critical.** At f = 0.99 the symmetric speedup with n = 256 is about 80 at
  best; at f = 0.999 it is about 200.
- **Using more BCEs per core can be optimal** (the best r is above 1 unless f ≈ 1), which is why cores remain
  fairly big. Denser chips make it more likely that the optimal core is non-minimal.
- **Asymmetric chips can beat symmetric ones considerably** (at n = 256, f = 0.975 the best symmetric is about 51
  and the best asymmetric about 125). This motivates single-ISA heterogeneity, with its scheduling and overhead
  challenges. Dynamic multicores (fusing cores) are best of all.

**Mobile processors:** battery energy density improves far more slowly than CPU speed, RAM or disk (Paradiso
2005). Design principles: at the high end, maximum compute within thermal bounds; at the low end, minimum power.

**Heterogeneous computing:** **functional heterogeneity** (MPSoCs with CPU, GPU, DSP and accelerators) versus
**performance heterogeneity** (the same ISA, different power and performance points: single-ISA heterogeneous or
asymmetric multicores). The latter addresses both Amdahl's serial bottleneck and dark-silicon energy efficiency.

**ARM big.LITTLE** (2011): **Cortex-A15** (complex OoO multi-issue, high performance) paired with **Cortex-A7**
(simple in-order 8-stage, energy-efficient); later A57/A53. Both implement the same ISA. A **GIC-400** interrupt
controller and the **CCI-400 cache-coherent interconnect** connect them; hardware coherence (ACE) makes task
migration between clusters cheap and transparent (snooping the other cluster's L2).

**Software execution models:**
1. **Cluster migration:** only one cluster is active; switch the whole cluster on load.
2. **CPU migration (in-kernel switcher):** pair each big core with a LITTLE core as one virtual CPU, with only
   one of each pair active.
3. **Global task scheduling (HMP):** all cores visible; the scheduler tracks each thread's compute needs and each
   core's load. This evolved into Linux **Energy-Aware Scheduling (EAS)** with per-entity load tracking (PELT)
   and an energy model, and ARM **DynamIQ** (big and LITTLE in one cluster).

These three are exactly Part 1 of my Assignment 3 (Part E).

## L17 (4 Apr) — Consistency and coherence for heterogeneous systems (Primer, 2nd ed., Ch. 10)

**Heterogeneous SoCs** combine CPU cores, GPU SMs and accelerators, each with private L1s and L2s, sharing an LLC
and memory controller. Two questions: how are the caches kept coherent, and what consistency model applies and
how is it enforced?

**Early GPUs:** relaxed consistency, no hardware coherence. They targeted workloads that neither synchronise nor
share data often. Consistency is **scoped**: with `FENCE`s around the MP test, if T1 and T2 are in the **same CTA**
(sharing an SM's L1), r2 = NEW. If they are in **different CTAs**, r2 can be 0, because the non-coherent L1 may
hold stale data. Scoped fences and atomics (`.cta`, `.gpu`, `.sys` in PTX; `__threadfence_block/_threadfence/
_system` in CUDA) and the **HRF** (heterogeneous-race-free) models formalise this.

**GPGPU workloads now synchronise and share finely.** But a CPU-style invalidation protocol would mean high traffic
and hardware cost (tracking sharers across thousands of threads). Hence **self-invalidation** schemes.

**Temporal coherence** (Singh et al., HPCA 2013): a self-invalidation protocol built on a **global notion of time**.
Assumptions: an **inclusive L2** and **write-through, no-write-allocate L1s**.
- A reader brings a block into L1 for a **lease**, a predicted time period. When the lease ends, the block
  self-invalidates; a read after expiry is a miss.
- L1 and L2 blocks are tagged with timestamps; L2 knows each block's latest lease.
- **Consistency-agnostic TC:** to enforce SWMR, a **write stalls at L2 until all leases on the block have
  expired**, so there are no sharers when it is performed. Program-order processors plus TC give **sequential
  consistency**.
  - L1 states I/V with messages `GetV(t)` (request a block for lease t), `Write` (write through without
    allocating) and `WriteV(t)`.
  - L2 states **I**, **P** (private: in exactly one L1), **S** (shared: possibly in several L1s) and **Exp**
    (present in L2 but expired everywhere). Writes to P or S blocks stall until expiry.
  - The worked timeline on the slide: data1 has lease 30, so a write issued at time 1 completes at 31. Flag has
    lease 35. Data2's lease (20) has already expired, so its write completes at once.
- **Consistency-directed TC:** stalling at L2 slows every thread. Instead, writes propagate **asynchronously** and
  only **FENCEs stall**. L2 replies to each write with a **GWCT** (global write completion time: when the write
  becomes visible to all, i.e. the latest lease expiry). Each SM keeps a per-thread **stall-time** register =
  max(GWCT). A FENCE waits until the current time ≥ stall-time. This enforces a **relaxed** (XC-like) model at
  much lower cost.
- **Limitations:** a non-inclusive L2 is hard to support; global timestamps are needed; performance depends on
  the chosen lease period; scoped synchronisation can't be exploited directly.
- Further reading: lazy release consistency for GPUs (MICRO 2016); relativistic cache coherence (SC on GPUs,
  HPCA 2017); formal PTX memory model (ASPLOS 2019); Atomic Cache (MICRO 2024); CPElide (multi-chiplet GPU
  implicit sync, MICRO 2024); over-synchronisation in GPU programs (MICRO 2024).

---

# Part D — Emerging computer architecture concepts

## L18 (9 Apr) — Processing-in-memory (PIM)

**Why:** von Neumann machines shuttle data between DRAM and the CPU/GPU. Many modern workloads (graph analytics,
databases, ML inference, especially LLM decode GEMV) are **fundamentally memory-bound**, and data movement
dominates latency and energy (a DRAM access costs orders of magnitude more energy than an add). PIM puts logic
**inside or near memory**. Motivations: data-movement bottlenecks, energy efficiency, emerging workloads,
memory–compute imbalance. A neural accelerator pulling all data from main memory has huge transfer energy; with
in-memory processing only the needed results cross the bus.

**Two approaches:**
1. **Processing-using-memory (PuM):** exploit the analog properties of memory cells (DRAM, SRAM, NVM) to compute,
   e.g. bulk bitwise operations across whole rows, or lookup-table-based operations.
   - **RowClone** (Seshadri et al., MICRO 2013) does **bulk copy and initialisation inside DRAM**. Two
     back-to-back ACTIVATEs within a subarray do it: activating row A latches it into the 4 KB (or 8 KB) row
     buffer, and activating row B while the buffer still drives the bitlines overwrites B with A's data. Copying
     a 4 KB page needs no transfer over the 8-bit-per-chip data bus, at negligible hardware cost. The paper
     reports about 11.6× lower latency and 74× lower energy for in-subarray copies.
   - **Ambit** (MICRO 2017) does bulk AND/OR via triple-row activation (charge sharing computes the majority) and
     NOT via dual-contact cells.
2. **Processing-near-memory (PnM, also near-data processing):** add compute to memory controllers, DIMM buffer
   chips, or the logic layer of 3D stacks (HMC, HBM). Examples: accelerators, reconfigurable logic, simple cores,
   functional units.
   - **Casper** (Denzler et al., IEEE Access 2023): a near-cache accelerator of specialised **stencil processing
     units** (SPUs) beside each LLC slice. Stencil data with little reuse skips the cache hierarchy, and its
     regular access patterns exploit SRAM-array parallelism (512-bit vector units).
   - **UPMEM** (the first commercial PIM): DDR4 DIMMs whose DRAM chips each contain **DPUs** (DRAM processing
     units). Each DPU is a **multithreaded in-order 32-bit RISC core** with **24 hardware threads** (24 × 32 GPRs
     each), a 24 KB IRAM and a **64 KB WRAM** scratchpad (8/16/32/64-bit load/store) beside a **64 MB MRAM** DRAM
     bank. The pipeline has 14 stages, and only the last three can overlap with DISPATCH/FETCH of the next
     instruction *from the same thread*, so at least 11 threads are needed to fill it. DMA instructions move code
     MRAM→IRAM and data MRAM↔WRAM. Programming is **SPMD in C**: software threads called **tasklets**, statically
     mapped, share MRAM/WRAM and synchronise with mutexes, barriers, handshakes and semaphores. A server holds up
     to about 2,560 DPUs. The host-to-DPU and inter-DPU paths go through the host, which is the key weakness
     (PrIM benchmarks: Gómez-Luna et al., IEEE Access 2022).
- **Applications:** AI/ML (matrix ops, inference), databases and search (filter, aggregate, index), scientific
  computing, edge/IoT. Industry has also built **Samsung HBM-PIM / Aquabolt-XL**, **SK hynix GDDR6-AiM** and
  **AxDIMM**, aimed at memory-bound GEMV in LLM inference.
- **Challenges:** programming models and code generation, runtime engines (adaptive code and data scheduling,
  mapping, access control), memory coherence between host and PIM, and virtual memory for a unified address
  space. (Primer: Mutlu et al., "A Modern Primer on Processing in Memory", arXiv 2012.03112.)

## L19 (11 Apr) — Approximate computing

**Motivation:** compute energy demand is exploding (data volume and AI) towards world energy-production limits,
while many applications are **error-resilient**. An image with a few wrong pixels looks the same to a human, and
ML classification survives imprecise parameters. Examples: a 5% accuracy drop lets k-means save up to 50% energy
(Chippa et al., TVLSI 2014); neural approximation speeds up inverse kinematics by 26% over GPU execution
(Grigorian & Reinman, TACO 2015). The idea is to **exploit the gap between the accuracy a system delivers and the
accuracy users need.**

**Approximation techniques by layer** (Xu et al., IEEE D&T 2016):

| Layer | Techniques |
|---|---|
| Program | loop perforation (skip iterations), code perforation, thread fusion, tunable kernels, pattern reduction |
| Architecture | approximate storage, ISA extensions, approximate accelerators |
| Circuit | imprecise logic, voltage over-scaling, analog computation, precision scaling |

**The paradigm:** **offline**, identify error-resilient parts, characterise approximate kernels and use
representative workloads. **Online**, evaluate runtime quality, tune approximation and manage quality
dynamically, all in a feedback loop.

**Architectures:**
- *Enhanced general-purpose processors:* annotate precise and approximate instructions or data (EnerJ-style type
  qualifiers). Fine-grained: ISA extensions, as in **Truffle** (Esmaeilzadeh et al., ASPLOS 2012), a
  **dual-voltage** pipeline where approximate instructions use low-V_dd functional units and register halves
  with level shifters, beside a precise path. Coarse-grained: offload approximate segments to efficient but
  unreliable cores (**Accordion**, HPCA 2014, soft near-threshold voltage). Gains are bounded by von Neumann
  overheads, which aren't approximated.
- *Algorithmic transformation:* **neural acceleration**, NPU (Esmaeilzadeh et al., MICRO 2012). Observe an
  approximable code region's inputs and outputs, train an MLP to mimic it, and replace the region with
  invocations of a neural processing unit.
- *Approximate memory and storage:* lower DRAM refresh for non-critical data (**Flikker**, ASPLOS 2011);
  lower-voltage SRAM; precision-aware soft-error protection for GPUs; approximate MLC storage in SSDs for longer
  lifetime and speed (Sampson et al., MICRO 2013); lossy memory-link compression for bandwidth.
- *Approximate arithmetic:* carry-skip adders with truncated carry chains; **accuracy-configurable adders** with
  carry-in prediction per segment and multiplexed correction.
- **Challenges:** limited application domain and gains, correctness, finding application-specific strategies,
  overhead and scalability, configurability for the required quality.

The ML connection is direct: **quantisation (INT8/FP8/FP4), pruning, sparsity and low-rank approximation** are
mainstream approximate computing.

## L20 (23 Apr) — Novel (non-volatile) memory technologies

**Why:** DRAM scaling has slowed (Mutlu, IMW 2013), DRAM accounts for more than 25% of datacenter energy
(Malladi et al., MICRO 2012), and big data pushes towards in-memory computing. Density, power, performance and
scalability all need work. Typical hierarchy: registers (~1000 B, 300 ps), L1 (64 KB, 1 ns), L2 (256 KB,
3–10 ns), L3 (2–4 MB, 10–20 ns), DRAM (4–16 GB, 50–100 ns), disk (4–16 TB, 5–10 ms).

**Memory array basics:** word lines (rows) and bit lines (columns), with a memory device plus an access device
(transistor or diode) at each crosspoint.

**Desirable properties of next-generation memory:** non-volatile, no moving parts, low cost. Targets (Qureshi et
al.): program time < 50–100 ns; programming energy < 1–10 pJ/bit; cell size < 10F²; 2–4 bits per cell; retention
> 10 years at 85 °C; endurance > 10⁸ cycles; CMOS-compatible processing (< 450 °C).

| Technology | Mechanism | Key points |
|---|---|---|
| **Flash** | floating-gate transistor with an electrically programmable threshold voltage; Fowler–Nordheim tunnelling or hot-carrier injection traps charge | NOR: random access, fast reads, slow writes, larger cells, code execution (XIP)/firmware. NAND: block access, sequential reads, faster writes, smaller and cheaper cells, data storage. Endurance about 10³–10⁵ |
| **FeRAM** | ferroelectric capacitor (1T-1C, or chain FeRAM) polarised by bit-line/plate-line pulses | endurance about 10¹², but **destructive reads** (must rewrite) |
| **STT-MRAM** | magnetic tunnel junction (free layer, barrier, fixed layer); current direction sets parallel (low R) or antiparallel (high R) | cell up to about 14F², endurance about 10¹², fast; candidate for LLC/embedded |
| **PCM** | chalcogenide (GST, Ge₂Sb₂Te₅) switched between amorphous (high R, "0", RESET = melt and quench) and crystalline (low R, "1", SET = anneal) by Joule heating | multi-level cells; **limited write endurance (~10⁸) and high write latency/energy** |

Comparison at a glance: Flash 27 nm (7 MB/s write, 200 Mb/s read, not byte-addressable except NOR reads); FeRAM
130 nm (83 ns write, 43 ns read); STT-RAM 54 nm (<15 ns write, <20 ns read); PCM 45 nm (100–500 ns write, 85 ns
read). FeRAM, STT-RAM and PCM are byte-addressable.

**Hybrid DRAM + NVM main memory:**
- **Horizontal:** DRAM and NVMM side by side under one hybrid memory controller, which migrates hot pages to
  DRAM. Limitations: costly page-level access monitoring and migration, and hot-page detection that is slow and
  inaccurate.
- **Hierarchical:** a small DRAM cache, invisible to the OS, in front of a large NVMM. Limitations: poor-locality
  applications suffer, and hardware-managed caches can't fully use DRAM capacity.
- **Intel Optane DC Persistent Memory** (3D XPoint, now discontinued) supports both. **Memory Mode** makes DRAM a
  direct-mapped cache of Optane, which appears as volatile main memory. **App Direct Mode** exposes Optane as
  persistent, byte-addressable memory (DAX). Internally there is a 256 B XPLine versus the 64 B cache line, so
  small random writes suffer write amplification (Yang et al., FAST 2020).
- **Persistent-memory management:** NVMM is both working memory and persistent storage. Correctness needs
  **write ordering and atomicity**: cache-line flushes (`clwb`) plus fences, logging (undo/redo), transactions,
  checkpoints, and failure-atomic data structures. Today **CXL-attached memory** (CXL.mem expansion and pooling)
  is the live successor to this tiering problem.

## L21 (25 Apr) — Chiplets, 3D integration and HBM

**Problems with monolithic chips:**
- **Yield** falls steeply with die area. Approximate dies per wafer ≈ π(R − √A)²/A (slide); H&P's form is
  π(d/2)²/A − π·d/√(2A). Die yield ≈ wafer yield × 1/(1 + D₀·A/N)^N (negative binomial, D₀ = defect density).
- **Reticle limit:** about 858 mm² maximum die size.
- **Scaling:** SRAM, analog and I/O no longer shrink with logic, so expensive leading-edge silicon is wasted on
  them.
- **Cost** per transistor has stopped falling at the newest nodes.
- **Thermal** limits.

Chiplets are the response. TSMC's roadmap projects more than 1 trillion transistors per package by about 2030
through 3D heterogeneous integration, versus about 200 billion monolithic.

**Chiplet architecture:** homogeneous (identical compute dies) or heterogeneous (mixing nodes: CPU at 5 nm, I/O
at 12 nm, memory, FPGA). Benefits: smaller dies yield better; no reticle limit, so huge systems are possible;
mature, cheaper nodes for I/O and analog; specialised processes for performance; disaggregation by function;
scalable product lines.

**Die-to-die interfaces:**

| Interface | Adoption | Package | Bump pitch | Lane rate | Latency | Reach | Energy | Edge density | Target |
|---|---|---|---|---|---|---|---|---|---|
| XSR/USR SerDes | industry standard | laminate | 130–170 µm | 112/224 Gb/s | ~10 ns | <50 mm | 1–4 pJ/bit | <3 Tb/s/mm | optical, networking |
| BoW (Bunch of Wires) | ~10 companies | laminate, fan-out, interposer | 45–170 µm | 2–32 Gb/s | <2 ns | <25 mm | 0.3–0.5 pJ/bit | <4 Tb/s/mm | disaggregation, AI accelerators |
| AIB (Intel) | 10+ third-party chiplets | fan-out, EMIB, interposer | 25–55 µm | 2–6.4 Gb/s | <2 ns | <2 mm | 0.5–0.85 pJ/bit | <1.6 Tb/s/mm | aerospace and defence |
| **UCIe** | 120+ member companies | laminate, EMIB, interposer | 10–130 µm | 4–32 Gb/s | <2 ns | <10 mm | 0.25–0.5 pJ/bit | <10 Tb/s/mm | scale-and-split system aggregation |

**Packaging:**

| Type | Description | Advantages | Applications |
|---|---|---|---|
| 2.5D | chiplets side by side on a **silicon interposer** (TSMC CoWoS) | high bandwidth, modular, good thermals | HPC, AI, datacenters |
| 3D | dies stacked with **TSVs** (and hybrid bonding, e.g. SoIC; AMD 3D V-Cache) | highest density, low latency, energy-efficient | HPC, AI, memory stacks |
| Fan-out wafer-level | chiplets in a reconstituted wafer with redistribution layers | low cost, high density, thin | mobile, IoT |
| EMIB (Intel) | small silicon bridges embedded in the substrate | compact, cheaper than full interposers | servers, datacenters |
| Package-on-package | memory stacked on the processor | compact, energy-efficient | mobile, wearables |

**Modern chiplet processors:** AMD EPYC 9965 (CCDs plus I/O die) and Intel Xeon 6980P (compute chiplets plus I/O
chiplets with EMIB).

**HBM:**
- **AMD Fiji** (2015) was the first GPU with HBM: 28 nm, 596 mm² die on a 1,011 mm² interposer, 4 HBM stacks,
  512 GB/s.
- **Cross-section:** stacked DRAM dies on a base logic die, connected by TSVs and micro-bumps. The stack and the
  GPU sit on a silicon interposer (short, dense wires), which sits on C4 bumps on the package substrate, which
  sits on balls on the board.
- **Generations:**

| Generation | Data rate (Gb/s per pin) | BW per stack (GB/s) | Stack height | Max DRAM per die (Gb) | Max capacity (GB) |
|---|---|---|---|---|---|
| HBM | 1.0 | 128 | 8 | 16 | 16 |
| HBM2 | 2.0 | 256 | 8 | 16 | 16 |
| HBM2E | 3.6 | 461 | 12 | 24 | 36 |
| HBM3 | 6.4 | 819 | 16 | 32 | 64 |
| HBM3E | 9.6 | 1,229 | 16 | 32 | 64 |

  Each stack has a **1024-bit** interface split into independent **channels** (8 in HBM2; 16 in HBM3) and
  **pseudo-channels**, each with its own banks. This massive parallelism is why AI accelerators use HBM. An H100
  SXM has 5 stacks at about 3.35 TB/s; a B200 has 8 HBM3E stacks at about 8 TB/s.

**The future of computer-architecture research** (closing slide): specialised AI/ML architectures, heterogeneous
computing, 3D integration and chiplets, memory-centric architecture, energy efficiency, security and reliability.

---

# Part E — My assignments

The folder holds Assignments 2 and 3. **Assignment 1** (the CPU-vs-GPU sine-wave simulation, Feb 2025) is only
partly here: `Assignments/Submitted_Assignments/Assignment_1/` has a 0-byte `CS24S009.zip` and an incomplete
`.zip.part` download. The full code lives in `~/MyProjects/RTR2021/03-OpenGL/02-PP/07-OpenGL-HPP/`. My assessment of
it as a project is in `Phase3_Project_Ideas.md` §12.10.

## Assignment 2 — Snooping MSI and MESI coherence in a 4-core simulator

**Task:** implement `Cache::handleProcRequest()` and `Cache::handleBusRequest()` in `coherence.cpp` for **both** MSI
and MESI (chosen at runtime). Submit it as `CS24S009.cpp`; grading is by test cases (2 public, the rest hidden).

**Simulator facts** (`simulator.h`, `bus.cpp`, `cacheset.cpp`, `cache.cpp`, `main.cpp`):
- 4 cores; each private cache is 4 sets × 4 ways with 64 B blocks (offset 6 bits, set 2 bits) and LRU. A set is a
  `std::list<CacheBlock>` with the front as LRU and the back as MRU. `insertCacheBlock` evicts the front and
  pushes to the back (MRU) and **returns the evicted block**. `setState(…, Invalid)` moves the block to the LRU
  position and zeroes its tag, so invalid ways are reused first.
- `Bus::sendMessage(req, block_addr, sender)` increments the per-type counters. For **Flush** it only counts
  (cache-to-cache transfer is disabled, so flushes go to memory). For BusRd/BusRdX it clears `shared_line` and calls
  `handleBusRequest` on every *other* cache. MESI's shared line is set by snoopers via `setSharedLine()` and read
  by the requester afterwards with `getSharedLine()`.
- Counters each cache must maintain: `num_reads`, `num_read_misses`, `num_writes`, `num_write_misses`,
  `num_writebacks` (flushes of dirty data, whether on eviction or due to snooping) and `num_invalidations`
  (blocks invalidated by the protocol).
- Input: protocol name, then lines `core r|w 0xaddr`, ending with `-1`.

**The correct transition table for this simulator:**

| Event | Action |
|---|---|
| PrRd hit (M, E or S) | `moveToMRU` |
| PrRd miss | `num_read_misses++`; `BusRd`; new state = MESI ? (shared ? S : **E**) : S; insert; if the evicted block was M → `Flush` + `num_writebacks++` |
| PrWr hit M | `moveToMRU` |
| PrWr hit E (MESI) | → M silently, `moveToMRU` |
| PrWr hit S | `BusRdX` (an upgrade; not counted as a miss), → M, `moveToMRU` |
| PrWr miss | `num_write_misses++`; `BusRdX`; insert as M; evicted M → `Flush` + writeback |
| Snoop BusRd in M | → S, `Flush`, `num_writebacks++`, assert shared |
| Snoop BusRd in E | → S, assert shared (**no flush: E is clean**) |
| Snoop BusRd in S | assert shared |
| Snoop BusRdX in M | → I, `Flush`, writeback++, invalidations++ |
| Snoop BusRdX in E or S | → I, invalidations++ (**no flush**) |

**What I submitted** (`CS24S009.cpp`, 19 Mar 2025): the structure matches the table, with one bug. In
`handleBusRequest`, BusRdX on an **E** block issues `Flush` and increments `num_writebacks`. E is clean, and the
assignment says flushes happen only when the block is dirty (my own lecture notes mark the diagram's E→I edge as
"Flush′", meaning only with cache-to-cache sharing).

**Verification I ran:**
- All three variants (`CS24S009.cpp`, `coherence.cpp`, `MSI_MESI.cpp`) **pass all 4 public tests** (MSI1, MSI2,
  MESI1, MESI2). The public tests never exercise "exclusive read, then another core writes".
- A targeted test (`MESI; 0 r 0x1000; 1 w 0x1000; 2 r 0x2000; 3 w 0x2000`) shows the submitted version producing
  **2 spurious flushes and 2 spurious writebacks** (correct answer: 0 and 0). Any hidden MESI test with that
  pattern fails.
- The later `coherence.cpp` (22 Mar) comments out that flush, so it's correct. But its eviction-flush line has
  a nonsense address, `evicted_block.tag<<SET_BITS||(blockAddress&&(SET_BITS)-1)` (logical `||`/`&&` instead of
  bitwise `|`/`&`). That's harmless here, because flush addresses aren't used, but it would be marked down in a
  code review.

**Phase-2 fix:** delete the E-state flush, compute the evicted block address properly
(`(tag << SET_BITS) | set`), and add a small self-written regression suite covering each transition.

## Assignment 3 — big.LITTLE scheduling (Clustering, Switcher, Global Task Scheduling, and my own "Bonus")

**Task:** implement `System::epoch()` in `scheduler.cpp` for four schemes, selected by `./sim <0..3> < input`.
Part 1 (7 marks) is test-case graded; Part 2 (8 marks) is the Bonus scheduler plus a report.

**Simulator semantics** (from the TA's `core.cpp`, `process.cpp`, `system.cpp`):
- Time advances in 1 ms ticks. Each tick, every process's `cycle` runs first (arrival and I/O countdown), then
  every core's `cycle`, then `epoch()` if `t % 100 == 0`. Processes arrive only at epoch boundaries.
- A core that is ON with no current process burns **idle power**, adds to idle time, and picks the next READY
  process (FCFS by lowest PID). A core running a process burns **running power**, adds 1 to load, and advances
  the process by `MIPS/1000` million instructions (LITTLE: 1 per ms, big: 4 per ms). A just-migrated process first
  pays `MIGRATION_PENALTY = 2` ms of running power with no progress.
- Load and energy history are recorded per core per epoch. **The metric is system-level
  EDP = (Σ core energy, mJ) × (total simulated time, ms).**
- Cores start **OFF**, and powering on or off is instantaneous and free. CPU bursts are non-preemptive except
  through migration.
- **Power model in the test inputs:** LITTLE 1000 MIPS, 1 W idle, 2 W running; big 4000 MIPS, 2 W idle, 6 W
  running. (The PDF says 2/4 W and 4/8 W; the README and test cases use the former.)

**Key insight for designing a scheduler.** Energy per million instructions while running is **LITTLE 2.0 mJ vs.
big 1.5 mJ** (2.0 vs. 4.0 under the PDF numbers). In this simulator a big core is **both 4× faster and cheaper
per instruction**. LITTLE cores win only while **idle** (1 W vs. 2 W). So the EDP-optimal strategy is roughly
"race to idle": run CPU work on big cores, keep as few cores powered as possible, consolidate I/O-heavy processes,
and power off any core with no live work. Classic intuition says "LITTLE is energy-efficient", but that is only
true when the big core's running power exceeds 4× the LITTLE core's.

**Part 1, as specified:** round-robin assignment of newly arrived processes to the *active* cores.
- **Clustering:** start with the LITTLE cluster (cores 0–3) ON. If the average LITTLE load over the last epoch is
  ≥ 50%, move everything to the big cluster (core i → i+4) and power the LITTLE cluster off. If the average big
  load is < 50%, move back.
- **Switcher:** four virtual cores (i, i+4). Switch each pair independently on its own last-epoch load (≥ 50% on
  LITTLE → big; < 50% on big → LITTLE).
- **Global:** all 8 cores ON, round-robin across all of them, no migration.

**Verification of my `scheduler.cpp` (24 Apr 2025)** against the TA's expected outputs for workloads 1 and 2:

| Scheme | workload1 | workload2 |
|---|---|---|
| 0 Clustering | **FAIL: never terminates**; prints millions of `[WARN]` lines (it filled 3.2 GB in under a minute) | **FAIL: never terminates** |
| 1 Switcher | PASS (EDP 83,955,072) | PASS (EDP 103,743,993) |
| 2 Global | PASS (EDP 150,187,940) | PASS (EDP 145,576,522) |

**Root causes in Clustering:**
1. New arrivals are placed only on a core whose `getScheduledProcesses()` is **empty**. Terminated processes are
   never removed from that list, so after the first wave no core is ever "empty". Arrivals are collected only when
   `getArrivalTime() == currentTime`, so an unplaced process is **lost forever**, and the simulation can't finish.
   In workload1 only PIDs 0–3 ever ran.
2. Every epoch it calls `powerOnCore()` on cores that are already ON, which prints `[WARN] Trying to power on an
   already powered on core` four times per epoch (an output diff failure even if the logic were right).
3. It isn't the specified plain round-robin.

The alternative `temp.cpp` terminates, but its Clustering EDP is wrong (83,485,024 vs. 83,614,762 for workload1;
143,019,529 vs. 105,095,394 for workload2), and its Switcher prints `[WARN]`s. **Neither version has a correct
Clustering scheduler.**

**My Bonus scheduler, and why it underperforms:** it classifies each *newly arrived* process by the sum of its past
CPU bursts (light ≤ 20 → LITTLE, heavy → big). But a newly arrived process **has no past bursts**
(`getPastBursts()` is empty while `currentBurstIndex == 0`). So **every process goes to a LITTLE core and the big
cores are never used.** It also re-powers cores redundantly and never migrates.

EDP (and average turnaround in ms) on all four workloads:

| Scheme | w1 IO-bound | w2 CPU-bound | w3 mix | w4 sporadic |
|---|---|---|---|---|
| Switcher | 83.96M (1400) | **103.74M** (935) | **101.43M** (1143) | 202.02M (954) |
| Global | 150.19M (1316) | 145.58M (1038) | 161.12M (1175) | 330.40M (1012) |
| My Bonus | **69.21M** (1469) | 134.24M (1923) | 113.01M (1699) | **198.92M** (1584) |

It wins on IO-bound and sporadic loads (LITTLE idle power is cheap), but loses 29% on CPU-bound and 11% on mixed
loads against the Switcher. No report PDF is present in the folder.

**Phase-2 plan:**
1. Rewrite Clustering exactly per spec: plain round-robin over active cores, collecting every `ARRIVED` process,
   and only powering on cores that are OFF. Diff it against `workload{1,2}-0.out`.
2. Design a real Bonus scheduler from the power-model insight: power-gate idle cores every epoch; place new
   arrivals on big cores by default; track a per-process **EWMA of completed CPU-burst lengths** and CPU/IO ratio
   (legal past information) to migrate I/O-dominated processes onto as few cores as possible (LITTLE if they're
   mostly waiting); add hysteresis to avoid migration ping-pong (each migration costs 2 ms of running power).
   Run a sensitivity study on the thresholds.
3. Write the report: rationale, sensitivity study, comparison table, and which workloads favour which scheduler.

---

# Part F — Revision questions (with short answers)

1. **Why does MESI need a shared line, and what does E buy?** To decide on a read miss whether to install the
   block in E (no other sharers) or S. E lets the first write proceed without a bus transaction, which helps
   private data.
2. **MESIF vs MOESI?** F designates one clean sharer to forward data (Intel). O lets a dirty block be shared
   without writing it back to memory, with the owner supplying data (AMD).
3. **Why can't snooping scale, and how does a directory fix it?** Broadcast bandwidth and snoop energy grow with
   core count. A directory sends point-to-point messages only to actual sharers, at the cost of directory storage
   and multi-hop latency.
4. **Give a TSO-allowed, SC-forbidden outcome.** The SB test (0, 0): both stores sit in store buffers while both
   loads read old values.
5. **Why are locked RMWs full fences on x86?** An atomic RMW must drain the store buffer before its load and hold
   coherence permission until its store, which orders it against all earlier and later memory operations.
6. **How does LL/SC avoid ABA?** Any write to the reserved address, even one restoring the same value, clears the
   reservation, so the SC fails.
7. **Why is X-Y routing deadlock-free?** It forbids every Y→X turn, so the channel dependency graph has no cycles.
8. **Wormhole vs virtual cut-through?** Both pipeline flits. VCT needs whole-packet buffers so a blocked packet
   collapses into one router; wormhole uses flit buffers, so a blocked packet spans several routers, causing HoL
   blocking that VCs mitigate.
9. **What is dark silicon, and what are the four responses?** Post-Dennard power limits mean only a shrinking
   fraction of transistors can switch at full speed. Responses: shrink, dim (spatial/NTV or temporal/sprinting),
   specialise, new circuit paradigms.
10. **What are the five DSA guidelines, and how does the TPU follow them?** Dedicated memories (UB, accumulators),
    more arithmetic (65K MACs), easiest parallelism (in-order SIMD/systolic), smaller types (INT8), a DSL
    (TensorFlow).
11. **Why are systolic arrays energy-efficient?** Each operand fetched from SRAM is reused across a row or column
    of MACs through cheap neighbour-to-neighbour links, so there are fewer memory and register-file accesses per
    MAC.
12. **Hill–Marty: why can asymmetric multicores beat symmetric ones?** The big core speeds up the serial fraction
    while many small cores keep parallel throughput high, attacking both terms of Amdahl's law.
13. **Temporal coherence in one sentence?** Readers hold blocks for a time lease and self-invalidate at expiry;
    writers (or, in the relaxed variant, fences) wait until all leases have expired, so no invalidation messages
    are needed.
14. **RowClone in one sentence?** Two back-to-back row activations in one subarray copy row A into row B through
    the row buffer, without moving data over the memory channel.
15. **Why chiplets?** Yield, reticle limits, non-scaling SRAM/IO/analog, and cost. Mix nodes, scale beyond one
    die, and reuse IP.
16. **What limits PCM as DRAM replacement?** Write endurance (~10⁸) and high write latency and energy. Hence
    hybrid DRAM-cache or page-migration designs, plus wear levelling.

---

# Part G — Bridges to AI infrastructure (what to build on)

| ACA topic | Where it shows up in LLM systems |
|---|---|
| Coherence and consistency, scoped GPU memory models | Correct multi-GPU and CPU-GPU synchronisation (CUDA memory scopes, fences in custom kernels such as all-reduce and FlashAttention's producer-consumer warp specialisation); unified-memory systems (Grace-Hopper NVLink-C2C, MI300A) |
| Interconnection networks (topology, routing, flow control, VCs) | Datacenter AI fabrics: NVLink/NVSwitch domains, InfiniBand/RoCE fat-trees and rail-optimised topologies, TPU ICI tori with optical circuit switches; collective algorithms (ring, tree, halving-doubling all-reduce) are routing and flow-control problems; congestion control for AI traffic |
| Dark silicon and DSAs, TPU systolic arrays | Every AI accelerator (TPU, Trainium, MTIA, Gaudi, Groq, Cerebras); tensor cores; roofline-based kernel design; why low precision (FP8/FP4) matters |
| DLP/GPU architecture | Kernel engineering: tiling for shared memory, occupancy, coalescing, TMA, warp specialisation, thread-block clusters (FlashAttention-3 on Hopper) |
| Heterogeneous scheduling (big.LITTLE, EAS) | Heterogeneous serving: prefill/decode disaggregation (DistServe, Splitwise) assigns compute-bound prefill and memory-bound decode to different hardware pools, the big.LITTLE idea at cluster scale; GPU power capping and energy-aware LLM serving (DynamoLLM) |
| Processing-in-memory | Memory-bound LLM decode (GEMV over weights and KV) is the poster workload for HBM-PIM and AiM; PIM-based attention offload (e.g. AttAcc, NeuPIMs) |
| Approximate computing | Quantisation (GPTQ, AWQ, FP8/FP4 training and inference), KV-cache compression, pruning and sparsity, early exit, speculative decoding (trading compute for latency) |
| NVM and hybrid memory, CXL | Tiered memory for KV caches and embeddings (HBM ↔ DDR ↔ CXL ↔ SSD offload, as in FlexGen, InfiniGen, LMCache); recommendation-model embedding tables |
| Chiplets and HBM | Accelerator packaging (B200 is two dies with NV-HBI; MI300X is chiplets on CoWoS); HBM capacity and bandwidth set the model-size and batch limits for serving |

Project seeds that use this course directly:
1. **An energy-aware heterogeneous LLM serving scheduler**, an Assignment 3 analogue at datacenter scale.
   Simulate (or run on 2 GPU types) prefill/decode routing that optimises an EDP-like metric (joules × latency
   SLO), validated on real traces (Azure LLM inference traces).
2. **A NoC/collective-communication simulator for all-reduce on different topologies**, extending L10 to compare
   ring, tree and hierarchical all-reduce under link failures and congestion.
3. **A cache-coherence-style KV-cache sharing protocol** for multi-instance serving (prefix sharing across
   replicas, with directories that track which node holds which KV blocks).

---

# Part H — References

- D. E. Culler, J. P. Singh, A. Gupta, *Parallel Computer Architecture: A Hardware/Software Approach*, 1998
  (Ch. 2 for L3; Ch. 5–8 for coherence and directories).
- J. L. Hennessy, D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th/7th ed. (Ch. 4 DLP,
  Ch. 5 TLP, Ch. 7 DSAs).
- V. Nagarajan, D. J. Sorin, M. D. Hill, D. A. Wood, *A Primer on Memory Consistency and Cache Coherence*,
  2nd ed., 2020 (L5, L6, L17).
- J. L. Hennessy, D. A. Patterson, "A New Golden Age for Computer Architecture", CACM 62(2), 2019.
- C. E. Leiserson et al., "There's Plenty of Room at the Top", *Science* 368, 2020.
- L. Lamport, "How to Make a Multiprocessor Computer that Correctly Executes Multiprocess Programs", IEEE TC 1979.
- S. V. Adve, K. Gharachorloo, "Shared Memory Consistency Models: A Tutorial", IEEE Computer 1996.
- P. Sewell et al., "x86-TSO: A Rigorous and Usable Programmer's Model for x86 Multiprocessors", CACM 2010.
- M. Herlihy, J. E. B. Moss, "Transactional Memory: Architectural Support for Lock-Free Data Structures", ISCA 1993.
- J. Mellor-Crummey, M. Scott, "Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors", TOCS 1991.
- W. J. Dally, B. Towles, *Principles and Practices of Interconnection Networks*, 2004; C. J. Glass, L. M. Ni,
  "The Turn Model for Adaptive Routing", ISCA 1992; W. J. Dally, "Virtual-Channel Flow Control", IEEE TPDS 1992.
- C. Kim, D. Burger, S. W. Keckler, "An Adaptive, Non-Uniform Cache Structure for Wire-Delay Dominated On-Chip
  Caches", ASPLOS 2002.
- B. M. Rogers et al., "Scaling the Bandwidth Wall", ISCA 2009.
- M. B. Taylor, "Is Dark Silicon Useful? Harnessing the Four Horsemen of the Coming Dark Silicon Apocalypse",
  DAC 2012; H. Esmaeilzadeh et al., "Dark Silicon and the End of Multicore Scaling", ISCA 2011.
- G. Venkatesh et al., "Conservation Cores", ASPLOS 2010; A. Raghavan et al., "Computational Sprinting", HPCA 2012.
- N. P. Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit", ISCA 2017.
- H. T. Kung, "Why Systolic Architectures?", IEEE Computer 1982.
- A. Putnam et al., "A Reconfigurable Fabric for Accelerating Large-Scale Datacenter Services", ISCA 2014;
  A. Caulfield et al., "A Cloud-Scale Acceleration Architecture", MICRO 2016.
- Y.-K. Choi et al., "In-Depth Analysis on Microarchitectures of Modern Heterogeneous CPU-FPGA Platforms", ACM TRETS 2019.
- M. D. Hill, M. R. Marty, "Amdahl's Law in the Multicore Era", IEEE Computer 2008.
- I. Singh et al., "Cache Coherence for GPU Architectures", HPCA 2013; D. R. Hower et al., "Heterogeneous-Race-Free
  Memory Models", ASPLOS 2014.
- V. Seshadri et al., "RowClone", MICRO 2013; V. Seshadri et al., "Ambit", MICRO 2017; O. Mutlu et al., "A Modern
  Primer on Processing in Memory", 2020–25; J. Gómez-Luna et al., "Benchmarking a New Paradigm: … a Real PIM
  System (UPMEM)", IEEE Access 2022; A. Denzler et al., "Casper", IEEE Access 2023.
- Q. Xu, T. Mytkowicz, N. S. Kim, "Approximate Computing: A Survey", IEEE D&T 2016; S. Mittal, ACM CSUR 2016;
  H. Esmaeilzadeh et al., "Architecture Support for Disciplined Approximate Programming (Truffle)", ASPLOS 2012;
  "Neural Acceleration for General-Purpose Approximate Programs", MICRO 2012.
- M. K. Qureshi, S. Gurumurthi, B. Rajendran, *Phase Change Memory: From Devices to Systems*, 2011; J. Yang et al.,
  "An Empirical Guide to the Behavior and Use of Scalable Persistent Memory", FAST 2020; H.-K. Liu et al., survey of
  NVM main memory, JCST 2021.
- Chiplets and HBM: UCIe 1.0/2.0 specifications; K. Asifuzzaman et al., "Demystifying the Characteristics of High
  Bandwidth Memory", 2021; JEDEC HBM3/HBM3E.
