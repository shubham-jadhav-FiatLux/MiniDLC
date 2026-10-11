# CS6600 — Computer Architecture (IIT Madras, Jul–Nov 2024) — Learning Notes

> One-file revision guide for CS6600. It covers (a) the theory the course and its three projects rest on,
> written lesson by lesson so it can be studied from scratch, (b) a precise description of each project,
> what I submitted, and how it was verified against the official reference outputs, and (c) a fix list
> for Phase 2 plus bridges from this material to AI-infrastructure work.
>
> **What the folder contains:** no lecture slides, only the three projects (spec PDFs, traces, reference
> outputs, Dockerfiles, my code and reports). The theory below is therefore reconstructed from the standard
> texts the course follows (Hennessy & Patterson; Shen & Lipasti) at the depth the projects demand.
> The projects are modelled on the well-known NCSU ECE 463/563 simulator projects (Eric Rotenberg's course):
> the cache + victim-cache simulator, the bimodal/gshare predictor simulator, and the Tomasulo-style
> out-of-order pipeline simulator. CPU/GPU-company interviewers recognise these immediately.

---

## Table of contents

1. Quantitative principles (iron law, Amdahl, Little, power/energy, EDP)
2. Memory hierarchy and caches (the Project 1 theory)
3. Pipelining fundamentals
4. Branch prediction (the Project 2 theory)
5. Instruction-level parallelism: static and dynamic scheduling (the Project 3 theory)
6. Thread-level parallelism: coherence, consistency, multithreading (overview; CS6400 goes deeper)
7. Data-level parallelism: vector, SIMD, GPU (overview; CS6023 goes deeper)
8. My projects: specification, what I built, verification, defects and Phase-2 fix list
9. Revision questions with answers
10. Bridges from CS6600 to AI infrastructure
11. References

---

## 1. Quantitative principles

### 1.1 The iron law of processor performance

```
CPU time = Instruction count × CPI × Clock period
         = (instructions / program) × (cycles / instruction) × (seconds / cycle)
```

| Factor | Mainly set by | Typical levers |
|---|---|---|
| Instruction count (IC) | ISA, compiler, algorithm | richer instructions, SIMD, better algorithms |
| CPI (or 1/IPC) | microarchitecture, workload behaviour | pipelining, superscalar issue, OoO, caches, branch prediction |
| Clock period | circuit technology, pipeline depth | deeper pipelines, better process nodes |

The factors trade against each other. A deeper pipeline shortens the clock period but raises the branch
misprediction penalty in cycles, which raises CPI. A CISC instruction lowers IC but may raise CPI.
**Only total time is the honest metric.** MIPS and GFLOPS are rates that can mislead across ISAs.

CPI decomposition (used throughout this course):

```
CPI = CPI_ideal + CPI_branch_stalls + CPI_memory_stalls + CPI_structural + ...
CPI_memory_stalls = (memory accesses / instruction) × miss rate × miss penalty (cycles)
CPI_branch_stalls = (branches / instruction) × misprediction rate × misprediction penalty
```

### 1.2 Amdahl's law

If a fraction `f` of execution time is sped up by a factor `s`:

```
Speedup = 1 / ((1 − f) + f / s)          upper bound as s → ∞ : 1 / (1 − f)
```

Example: 90% of runtime parallelised on 100 cores → 1/(0.1 + 0.009) ≈ 9.2×, not 100×.
Corollary: **make the common case fast**, and remember that the un-accelerated part dominates later.
(Gustafson's law is the scaled-workload counterpoint: with problem size growing with p, speedup ≈ (1−f) + f·p.)

### 1.3 Little's law (the most useful law for sizing hardware queues)

```
Average occupancy L = arrival rate λ × average residence time W
```

Applications:
- **Instruction window / ROB / issue queue:** to sustain IPC = N when each instruction spends W cycles in the
  window (waiting for operands + executing), you need about N × W entries. Section 5.10 shows this directly
  in my Project 3 data: the scheduling-queue size at which IPC saturates grows in proportion to N.
- **Memory-level parallelism:** to sustain bandwidth B with latency T, you need B × T bytes in flight
  (e.g. MSHR count × line size ≥ B × T).
- **GPUs:** occupancy (resident warps) must cover latency × issue rate. Same law, different hardware (see CS6023).

### 1.4 Power and energy

- Dynamic power: `P_dyn = α · C · V² · f` (α = activity factor, C = switched capacitance).
  Dynamic energy per operation is `∝ C · V²`, independent of f.
- Static (leakage) power: `P_static = V · I_leak`. It grows with temperature and with lower threshold voltages.
- **Dennard scaling** (constant power density as transistors shrink) ended around 2005–2007, because V could
  no longer scale down. Consequences: the multicore turn, "dark silicon", and the move to **specialised
  accelerators** (GPUs, TPUs). This is the historical reason AI hardware looks the way it does.
- Combined metrics: Energy × Delay (EDP) and Energy × Delay² (ED²P) weight performance against energy.
  Minimising energy alone favours a very slow machine; minimising delay alone ignores the energy bill.
  EDP finds a balance. Project 1 uses EDP.

### 1.5 Simulation methodology (relevant to all three projects)

- **Trace-driven simulation:** replay a recorded stream of events (memory addresses, branch outcomes,
  instruction dependences) through a model. Fast and deterministic, but it cannot model wrong-path effects
  or timing feedback on the trace itself. All three projects are trace-driven.
- **Execution-driven / cycle-accurate simulators:** gem5, ChampSim (cache/prefetch research),
  Sniper, GPGPU-Sim / Accel-Sim (GPUs).
- **Validation by diffing** against a reference simulator is standard practice. Always match the reference's
  exact output format before studying the design space.

---

## 2. Memory hierarchy and caches

### 2.1 Why a hierarchy

The processor–memory performance gap (the "memory wall") means a DRAM access costs roughly 100–300 cycles.
Rough modern latencies: L1 about 1 ns (4–5 cycles), L2 about 3–5 ns, LLC about 10–20 ns, DRAM about 60–100 ns,
NVMe SSD about 10–100 µs. Small, fast memories near the core hold the working set; large, slow ones back them.
This works because of **locality**:
- **Temporal locality:** a recently used item is likely to be used again soon (loops, stack, hot variables).
- **Spatial locality:** items near a recently used item are likely to be used soon (arrays, sequential code).

### 2.2 Cache organisation: blocks, sets, ways

A cache of `SIZE` bytes, `BLOCKSIZE` bytes per block and associativity `ASSOC` has:

```
#blocks = SIZE / BLOCKSIZE
#sets   = SIZE / (ASSOC × BLOCKSIZE)
offset bits = log2(BLOCKSIZE)
index bits  = log2(#sets)
tag bits    = address bits − index bits − offset bits       (32-bit addresses in Project 1)
```

| Address field | Purpose |
|---|---|
| tag (upper bits) | identifies which memory block occupies a way |
| index (middle bits) | selects the set |
| offset (lower bits) | selects the byte within the block |

Worked examples using the Project 1 validation configurations:

| Config | #sets | offset | index | tag |
|---|---|---|---|---|
| L1 1024 B, 2-way, 16 B | 1024/(2·16) = 32 | 4 | 5 | 23 |
| L1 1024 B, direct-mapped, 16 B | 64 | 4 | 6 | 22 |
| L2 8192 B, 4-way, 16 B | 128 | 4 | 7 | 21 |
| VC 16 blocks (fully associative), 16 B | 1 | 4 | 0 | 28 |

In a simulator, never manipulate addresses as binary strings. Use integer shifts and masks:
`block = addr >> offset_bits; set = block & (sets − 1); tag = block >> index_bits`.
For a fully associative structure, store and compare the **block address** (`addr >> offset_bits`), not a
per-set tag. Forgetting this broke my victim cache (Section 8.1).

### 2.3 Placement: direct-mapped vs set-associative vs fully associative

| Organisation | Hit time / energy | Conflict misses | Hardware |
|---|---|---|---|
| Direct-mapped (1-way) | lowest; no way-select mux | highest | one tag comparator |
| A-way set-associative | moderate | fewer as A grows | A comparators + way mux |
| Fully associative | highest; CAM search | none (only compulsory and capacity) | comparator per block |

Rule of thumb (H&P): a direct-mapped cache of size X has about the same miss rate as a 2-way cache of size
X/2, and returns diminish beyond 8 ways.

### 2.4 Replacement policies

- **LRU (least recently used):** evict the block untouched for the longest time. Exact LRU needs per-set
  ordering state (log2(A!) bits). Simulators keep a per-block timestamp or counter, or keep each set as an
  MRU→LRU ordered list. Project 1 requires printing each set in **MRU-to-LRU order**, so the list
  representation maps naturally onto the output.
- **Pseudo-LRU (tree-PLRU):** A−1 bits per set form a binary tree pointing towards the "older" half. Hardware
  uses this for 8–16 ways.
- **FIFO, random:** cheap. Random behaves surprisingly well at high associativity.
- **RRIP / SRRIP / DRRIP (Jaleel et al., ISCA 2010):** predicts re-reference intervals, so streaming data
  doesn't thrash the cache. Modern LLCs use RRIP-like or learned policies (Hawkeye, Mockingjay).
- **Belady's OPT (MIN):** evict the block whose next use is furthest in the future. It's unrealisable but is
  the upper bound used to judge policies.

### 2.5 Write policies

| | Write-through (WT) | Write-back (WB) |
|---|---|---|
| Every write goes to the next level | yes (usually through a write buffer) | no; the block is marked **dirty** |
| Eviction | clean, no traffic | dirty victims are written back |
| Pros | simple coherence, next level always current | far less traffic |

| | Write-allocate (WA) | No-write-allocate (NWA) |
|---|---|---|
| A write miss | fetches the block and then writes it | writes around the cache |
| Typical pairing | write-back | write-through |

Project 1 uses **WBWA**: a write hit sets the dirty bit; a write miss allocates the block (reading it from
the next level), then writes it and marks it dirty; a dirty victim produces a writeback to the next level.

### 2.6 Classifying misses: the 3 Cs (plus coherence)

- **Compulsory (cold):** the first reference to a block. It would miss even in an infinite cache.
- **Capacity:** the working set exceeds the cache. These are misses a fully associative cache *of the same
  size* still takes, minus the compulsory ones.
- **Conflict:** too many blocks map to one set. These are misses the set-associative cache takes that a fully
  associative cache of the same size would not.
- **Coherence (4th C, multiprocessors):** the block was invalidated by another core's write.

How to measure them from simulation (needed for Project 1, Plot #1):

```
compulsory miss rate  ≈ miss rate of a very large fully associative cache (the flat floor of the curves)
capacity  (size S)    = MR(FA, S) − compulsory
conflict  (size S, A) = MR(A-way, S) − MR(FA, S)
```

In Plot #1 (L1 miss rate against log2(SIZE), one curve per associativity) the FA curve is the
capacity + compulsory envelope. The vertical gap between each A-way curve and the FA curve at a given size
is that associativity's conflict miss rate. All curves flatten at the compulsory floor as size grows.

### 2.7 Average memory access time (AMAT, called AAT in the project)

Single level:
```
AMAT = Hit time + Miss rate × Miss penalty
```

Two levels:
```
AMAT = HT_L1 + MR_L1 × (HT_L2 + MR_L2(local) × Miss penalty_L2)
```

- **Local miss rate** of L2 = L2 misses / L2 accesses. It looks large, because L1 has already filtered out
  the easy accesses.
- **Global miss rate** of L2 = L2 misses / CPU accesses = MR_L1 × MR_L2(local). This is what reaches DRAM.

**Exact formulas used by the Project 1 reference simulator.** I verified these by reproducing all eight
reference AATs to four decimals:

```
Miss penalty (ns)  MP = 20 ns + BLOCKSIZE / (16 GB/s) = 20 + BLOCKSIZE/16   (16 GB/s = 16 B/ns)
                        → 16 B blocks: 21 ns, 32 B blocks: 22 ns
SRR               = swap requests / (L1 reads + L1 writes)
MR_{L1+VC}        = (L1 read misses + L1 write misses − swaps) / (L1 reads + L1 writes)
MR_L2             = L2 read misses / L2 reads        (writebacks into L2 don't stall the CPU)

No L2   : AAT = HT_L1 + SRR × HT_VC + MR_{L1+VC} × MP
With L2 : AAT = HT_L1 + SRR × HT_VC + MR_{L1+VC} × (HT_L2 + MR_L2 × MP)
```

CACTI hit times implied by the reference outputs (45 nm): HT(1 KB DM, 16 B) ≈ 0.1289 ns;
HT(1 KB 2-way, 16 B) ≈ 0.1748 ns; HT(16-block FA VC, 16 B) ≈ 0.1404 ns; HT(8 KB 4-way, 16 B) ≈ 0.2311 ns.

Example (reference run 3: L1 1 KB 2-way + 16-entry VC + L2 8 KB 4-way):
`0.1748 + 0.1594×0.1404 + 0.1314×(0.2311 + 0.4529×21) = 1.4777 ns` ✔

### 2.8 Victim caches (Jouppi, ISCA 1990)

**Motivation:** direct-mapped caches are fast but suffer conflict misses when a few hot blocks map to the
same set and ping-pong. A small fully associative buffer holding recently **evicted** blocks catches most of
those ping-pongs at a fraction of the cost of extra associativity. Jouppi showed that a handful of entries
removes a large share of a direct-mapped L1's conflict misses.

**Protocol as specified in Project 1.** L1 and its VC interact only on an L1 miss:
1. If the L1 set has an invalid way, the set has never evicted anything, so the VC cannot hold the block.
   Skip the VC and go to the next level.
2. Otherwise the set is full. Pick the L1 LRU victim V. Issue a **swap request** to the VC (count it).
   - **VC hit on X:** swap X and V. X moves into L1 (carrying its dirty bit; it becomes MRU in L1). V moves
     into the VC, taking X's slot and becoming MRU in the VC. Count a **swap**. The next level is not touched.
   - **VC miss:** the VC makes room for V by evicting its LRU block. If that block is dirty, it is written
     back to the next level and counted as an L1/VC writeback. V enters the VC as MRU. Then L1 reads X from
     the next level, installs it as MRU, and marks it dirty if the request was a write.
3. The VC is fully associative, uses LRU, has the same block size as L1, and is matched on **block address**.

Statistics: swap request rate (SRR) = swap requests / accesses. The combined L1+VC miss rate subtracts swaps
from L1 misses. A swap costs about one VC access, versus a full next-level access for a real miss.

Energy accounting in the spec: every swap request touches the VC twice (search for X, then insert V),
so it contributes `2 × swap requests × E_VC`.

### 2.9 Multi-level caches and inclusion

| Policy | Invariant | Pros | Cons |
|---|---|---|---|
| Inclusive | L2 ⊇ L1 | an L2 lookup can filter coherence snoops | wasted capacity; L2 eviction must back-invalidate L1 |
| Exclusive | L1 ∩ L2 = ∅ | effective capacity = L1 + L2 (AMD) | more data movement (swaps) |
| **NINE** (non-inclusive, non-exclusive) | none enforced | simple | used in Project 1 |

In Project 1 (NINE), a block missing in both L1 and L2 is allocated in both. An L2 eviction doesn't
back-invalidate L1. When block allocation is implemented generically ("on a miss, make room; write back a
dirty victim to the next level; then read the requested block from the next level"), cascading L1→L2→memory
falls out automatically. **Ordering matters:** issuing the victim's writeback to L2 before or after the read
of X changes L2's LRU state, and therefore the final contents. The ECE 563-style reference writes the
victim back first, then reads the requested block. Validate this ordering against the reference outputs.

Traffic sanity checks given in the spec:
- with L2: memory traffic = L2 read misses + L2 write misses + L2 writebacks
- without L2: memory traffic = L1 read misses + L1 write misses − swaps + L1/VC writebacks

### 2.10 CACTI, energy and area

**CACTI** (HP Labs; Wilton & Jouppi 1996; CACTI 6.0 by Muralimanohar, Balasubramonian and Jouppi) models
SRAM/cache arrays. Given size, block size, associativity, technology node and ports, it estimates access
time, dynamic energy per access, leakage and area. Project 1 calls it through `popen("./cacti SIZE BLOCK
ASSOC 45nm 1")` (or `FA` for fully associative) via the provided `parse.h`.

**Always check its return value.** Tiny structures, such as small VCs, can make CACTI fail. The spec's
fallback is HT_VC = 0.2 ns.

Energy model (no L2):
```
E = (L1 reads + L1 writes)·E_L1          // every access probes L1
  + (L1 read misses + L1 write misses)·E_L1   // filling L1 on a miss
  + 2·(swap requests)·E_VC
  + (L1 misses − swaps)·E_mem + (L1/VC writebacks)·E_mem       E_mem = 0.05 nJ
```
With L2, add `(L2 reads + L2 writes)·E_L2 + (L2 misses)·E_L2` and charge memory for L2 misses and L2
writebacks instead.

**EDP = total energy × total access time**, where total access time = AAT × (L1 reads + L1 writes).
The reference prints it in nJ·ns. I confirmed this on reference run 0: E_total ≈ 1453.7 nJ,
T = 3.5352 ns × 100,000, which implies E_L1 ≈ 0.00189 nJ for a 1 KB 2-way array.
Area = sum of the CACTI areas (mm²) of L1, plus L2 if enabled, plus the VC if enabled.

### 2.11 The design-space questions Project 1 asks, and what to expect

1. **Miss rate vs size and associativity (Plot 1):** miss rate falls with size, with the knee where the
   working set fits. Associativity helps most at small sizes, where conflicts dominate. All curves converge
   at large sizes (the compulsory floor).
2. **AAT vs size and associativity (Plot 2):** the miss-rate gain from size and associativity competes with
   the rising CACTI hit time. The optimum is an interior point: big enough to capture the working set,
   not so big or highly associative that hit time dominates.
3. **Adding an L2 (Plot 3):** the miss penalty seen by L1 falls from about 22 ns to HT_L2 + MR_L2 × 22.
   The best L1 shifts *smaller*, since L1 misses are now cheaper, and the AAT optimum improves substantially.
   Compare EDP and area: L2 adds area but cuts energy-expensive DRAM traffic.
4. **Block size (Plot 4):** larger blocks exploit spatial locality (fewer compulsory misses) but cause
   pollution and fewer sets (more conflict and capacity misses). Small caches suffer pollution first, so
   their optimal block size is smaller; larger caches tolerate larger blocks.
5. **L1/L2 co-optimisation (Plots 5–6):** AAT and EDP surfaces over (L1 size, L2 size) with L1 < L2.
   Look for flat regions where a much cheaper (smaller-area) configuration is within 5% of the best.
6. **Victim cache (Plot 7):** a direct-mapped L1 with a VC often approaches 2-way at the same size, and
   keeps the faster DM hit time. Larger VCs show diminishing returns.

### 2.12 Advanced cache optimisations (H&P Chapter 2: the "ten optimisations")

| Goal | Technique | Idea |
|---|---|---|
| Lower hit time | small, simple L1; way prediction | keep L1 tiny and low-associativity; predict the way to read |
| Higher bandwidth | pipelined caches; multibanked caches | overlap accesses; interleave banks by address |
| Higher bandwidth | **non-blocking (lockup-free) caches** | keep serving hits under misses; **MSHRs** track outstanding misses (Kroft 1981) |
| Lower miss penalty | critical word first / early restart | return the requested word first |
| Lower miss penalty | merging write buffer | coalesce writes to the same block |
| Lower miss rate | compiler optimisations | loop interchange, **blocking/tiling** (the basis of GEMM tiling on CPUs and GPUs) |
| Lower miss penalty/rate | hardware prefetching | next-line, stride, stream buffers (also Jouppi 1990) |
| Lower miss penalty/rate | software prefetching | prefetch instructions inserted by the compiler or programmer |

### 2.13 Virtual memory, TLBs and caches

- Page tables translate virtual to physical addresses. A **TLB** caches translations. A page walk on a TLB
  miss costs several memory accesses; on x86-64 that's 4–5 levels, which page-walk caches mitigate.
- **VIPT L1 caches** index with untranslated page-offset bits while the TLB translates in parallel. To avoid
  synonyms: `cache size / associativity ≤ page size`. With 4 KB pages, that's why L1D sizes are 32 KB × 8-way
  or 48 KB × 12-way.
- **Huge pages** (2 MB or 1 GB) increase TLB reach. This matters a great deal for large ML/HPC footprints.

### 2.14 DRAM essentials

DRAM chips are organised as channels → ranks → banks → rows/columns. An access opens a row into the row
buffer (tRCD), reads columns (tCAS), and closes it (tRP). Row-buffer hits are cheap. Bandwidth is set by
channels × bus width × transfer rate. **HBM** stacks DRAM dies next to the processor over thousands of wires
(over 3 TB/s per GPU on H100 and later). This is the key resource for LLM inference (Section 10).

---

## 3. Pipelining fundamentals

### 3.1 The classic 5-stage RISC pipeline

`IF → ID → EX → MEM → WB`. Pipeline registers separate the stages. Ideal CPI is 1, and the clock period is set
by the slowest stage plus latch overhead.

### 3.2 Hazards

| Hazard | Cause | Remedies |
|---|---|---|
| Structural | two instructions need one resource in the same cycle | duplicate or pipeline the resource; stall |
| Data RAW (true dependence) | a consumer reads before the producer writes | **forwarding/bypassing**; stall (load-use: one bubble in a 5-stage pipe) |
| Data WAR (anti) | a later instruction writes before an earlier one reads | can't happen in an in-order 5-stage pipe; **register renaming** in OoO |
| Data WAW (output) | two writes reach the destination in the wrong order | in-order write-back; renaming |
| Control | the next PC is unknown until a branch resolves | **branch prediction**, delayed branches, predication |

`Pipeline CPI = 1 + structural stalls + data stalls + control stalls (per instruction)`.

### 3.3 Branch penalty arithmetic

With a branch frequency of 20% and a 3-cycle penalty on every branch, CPI = 1 + 0.2 × 3 = 1.6, a 37.5%
slowdown. Deeper and wider machines multiply the penalty: a 15–20 stage, 4–8-wide core loses tens of issue
slots per misprediction. That is why prediction accuracy matters so much (Section 4).

---

## 4. Branch prediction

### 4.1 Static prediction

Always-taken; BTFN (backward-taken, forward-not-taken, which is right for loops); profile-guided hints.

### 4.2 Dynamic prediction with saturating counters (Smith, ISCA 1981)

- **1-bit predictor:** remember the last outcome. A loop branch mispredicts twice per loop execution
  (at the exit, then again at the next entry).
- **2-bit saturating counter:** states 0 (strongly not-taken), 1 (weakly not-taken), 2 (weakly taken) and
  3 (strongly taken). Predict taken if the counter is 2 or more. Increment on taken (saturating at 3) and
  decrement on not-taken (saturating at 0). The hysteresis means a loop branch mispredicts only once per
  loop execution. The project initialises every counter to 2 (weakly taken).

### 4.3 Bimodal predictor (the project with n = 0)

A table of 2^m counters indexed by `PC[m+1:2]`, i.e. `(PC >> 2) & (2^m − 1)`. The low two bits are dropped
because instructions are 4-byte aligned. **Aliasing:** different branches sharing an entry. It is
destructive when they disagree. Larger tables reduce aliasing until the static branch working set fits,
after which accuracy saturates.

### 4.4 Correlating and two-level predictors

Branches correlate with other recent branches (e.g. `if (a==2) ...; if (b==2) ...; if (a!=b) ...`).
- **Global history register (GHR):** a shift register of recent outcomes.
- **Two-level adaptive prediction (Yeh & Patt, MICRO 1991):** the first level is history (global, per-address
  or per-set); the second level is pattern tables of counters. The taxonomy is GAg, GAs, GAp, PAg, PAs, PAp,
  and so on.
- **gselect** concatenates PC bits with history bits. **gshare** (McFarling, DEC WRL TN-36, 1993) XORs them.
  XOR spreads (branch, history) pairs across the whole table with less aliasing for the same size.

### 4.5 gshare exactly as specified in the project

```
pc_index = (PC >> 2) & ((1 << m) − 1)               // m low-order PC bits (bits m+1..2)
index    = pc_index XOR (GHR << (m − n))           // n-bit GHR XORed with the UPPER n bits of pc_index
predict  = counter[index] >= 2
update counter with the actual outcome (saturating)
GHR      = (GHR >> 1) | (outcome << (n − 1))       // shift right, newest outcome enters the MSB
```

Everything is initialised as GHR = 0 and all counters = 2. With n = 0 this reduces to bimodal.

### 4.6 Storage-budget arithmetic (the report got this wrong)

Each entry is a 2-bit counter. A 16 KB budget is 16 × 1024 × 8 = **131,072 bits**:
```
2^m × 2 ≤ 131,072  ⇒  2^m ≤ 65,536  ⇒  m ≤ 16
```
(The GHR's n bits are negligible.) My report wrote "2^m ≤ 8192 ⇒ m ≤ 13". That is wrong by a factor of
8 (it divided by 16 instead of 2), and it led me to stop the design-space exploration far too early.

### 4.7 What the data actually says

I re-ran my own simulator, which is numerically correct, for the full legal range:

Bimodal misprediction rate:

| m | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| gcc | 26.65 | 22.43 | 18.49 | 15.67 | 13.65 | 12.47 | 11.72 | 11.37 | 11.30 | 11.21 |
| jpeg | 7.92 | 7.79 | 7.74 | 7.70 | 7.62 | 7.60 | 7.59 | 7.59 | 7.59 | 7.59 |

Best gshare n for each m (all n ≤ m searched):

| m | gcc best (n, rate) | jpeg best (n, rate) |
|---|---|---|
| 12 | n=1, 12.04% | n=8, 6.84% |
| 13 | n=7, 10.56% | n=8, 6.83% |
| 14 | n=6, 9.08% | n=8, 6.69% |
| 15 | n=7, 8.20% | n=14, 6.67% |
| 16 | n=9, **7.49%** | n=14, **6.57%** |

Interpretation, which is what a good report says:
- **gcc** has a large static-branch footprint (a compiler). Bimodal keeps improving until about m = 13–14,
  then saturates at about 11.2%, the irreducible bias-only error. gshare at *small* m is **worse** than
  bimodal (m=7, n=2 gives 28.98% against 26.65%). History spreads each branch over up to 2^n entries,
  which multiplies aliasing and training time in a small table. Once the table is large (m ≥ 13),
  correlation pays off strongly: 7.49% against 11.21%, a 33% relative reduction. So for gcc within 16 KB,
  choose gshare (m=16, n≈9), or gshare (m=14, n=6) at 4 KB as the cost-efficient knee.
- **jpeg** has few, highly biased branches (tight loops). Bimodal saturates by m ≈ 10–11 at about 7.6%, so a
  256-byte bimodal table is enough. gshare with long history (n = 8–14) captures loop-exit and
  data-dependent correlation, reaching about 6.6–6.8%. The cost-effective gshare is (m=12, n=8) at 1 KB.
- My report proposed gcc gshare (m=10, n=4) at 17.99% and jpeg gshare (m=9, n=6) at 8.13%. Both are far
  from the knee, and the second is worse than a same-size bimodal.

### 4.8 Beyond gshare (good interview depth)

- **Tournament / combining predictors** (McFarling 1993; Alpha 21264): a chooser table selects between a
  local and a global predictor per branch.
- **Bi-mode, agree, YAGS:** reduce destructive aliasing by separating taken-biased and not-taken-biased
  branches.
- **Perceptron predictor** (Jiménez & Lin, HPCA 2001): a linear model over long histories, with history
  length growing linearly in hardware cost. It was the first ML-based predictor.
- **TAGE** (Seznec & Michaud, JILP 2006): several tagged tables indexed with **geometrically increasing
  history lengths**, where the longest matching history provides the prediction. TAGE-SC-L won the CBP
  championships and is the basis of modern high-end predictors.
- **BTB** (branch target buffer) predicts the target of taken branches; the **RAS** (return address stack)
  predicts returns; **ITTAGE** predicts indirect targets.
- Metric: **MPKI** (mispredictions per 1000 instructions) matters more than the misprediction rate,
  because it maps directly to lost cycles.

---

## 5. Instruction-level parallelism: static and dynamic scheduling

### 5.1 Dependences

- **True (data, RAW):** a consumer needs the producer's value. This is the fundamental limit.
- **Name dependences:** anti (WAR) and output (WAW). They arise from reusing a register *name*, so
  **register renaming removes them**.
- **Control dependences:** an instruction executes only if a branch goes a certain way. Speculation removes
  them; prediction plus recovery handle the cost.

### 5.2 Static (compiler) scheduling

Loop unrolling (fewer branches, more independent instructions, at the cost of registers and code size),
instruction scheduling to cover latencies, **software pipelining** (modulo scheduling), and VLIW/EPIC (Itanium),
where the compiler packs independent operations into wide bundles. VLIW's weakness is unpredictable latencies
(cache misses), which static schedules can't adapt to.

### 5.3 Scoreboarding (CDC 6600, 1964)

Instructions issue in order, read operands when available, execute out of order and write results. The
scoreboard tracks functional-unit status and register result status. It stalls on WAR and WAW because there is
no renaming, and that is the key limitation Tomasulo's algorithm removes.

### 5.4 Tomasulo's algorithm (IBM 360/91 FPU, Tomasulo 1967)

- **Reservation stations (RS)** sit in front of the functional units. Each entry holds the op, the source
  values `Vj, Vk` (if ready) or the producing RS tags `Qj, Qk` (if pending), and busy bits.
- **Register status table** (`Qi` per register) records which RS will produce the register's next value.
  This is the renaming: the architectural register name is replaced by a producer tag.
- **Issue (dispatch):** take the next instruction in order if an RS is free. Read ready source values, or copy
  the producer tags. Set the destination register's `Qi` to this RS's tag. WAR and WAW disappear because later
  readers get the tag of the *latest* producer and earlier readers already hold their values or tags.
- **Execute:** when both operands are present, start on the functional unit.
- **Write result:** broadcast (tag, value) on the **common data bus (CDB)**. Every RS waiting on that tag
  captures the value (**wakeup**). The register file updates only if its `Qi` still matches, i.e. no newer
  producer has been dispatched. Exactly this "update only if the tag still matches" rule appears in my
  Project 3 code.

### 5.5 Speculation, the ROB and precise exceptions

Tomasulo alone completes out of order, which gives **imprecise exceptions** (Project 3's model explicitly
accepts this). Adding a **reorder buffer (ROB)** (Smith & Pleszkun 1985) lets instructions complete out of
order but **commit in order**. Results sit in the ROB until commit. On a branch misprediction or exception,
everything younger in the ROB is flushed. This gives speculation past branches with precise state, which is
the basis of every modern OoO core.

### 5.6 Modern out-of-order cores

- **Unified physical register file + RAT + free list** (MIPS R10000, Yeager 1996). Rename maps architectural
  registers to physical ones; commit frees the *previous* mapping.
- **Issue queue wakeup/select:** CAM-based tag broadcast for wakeup, and an age-ordered or position-based
  select of up to N ready instructions per cycle. Project 3's "issue oldest ready first, up to N" is exactly
  an age-based select.
- **Load/store queue (LSQ):** memory disambiguation, store-to-load forwarding and memory-dependence
  prediction (store sets; Chrysos & Emer 1998).
- **Front end:** fetch bandwidth, fetch breaks at taken branches, trace caches and uop caches, decoupled
  front ends with fetch target queues.
- **Limits of ILP** (Wall 1991): window size, branch prediction, renaming registers and memory alias analysis
  bound the achievable ILP. Real code rarely sustains more than 2–4 IPC.

### 5.7 The Project 3 machine

N-wide fetch, dispatch and issue. The dispatch queue holds 2N, the scheduling queue (a unified RS pool) holds
S, and there are N universal, fully pipelined FUs. Latencies are 1 cycle for type 0, 2 for type 1 and 10 for
type 2. Caches and branch prediction are perfect, there is no ROB-based commit, and exceptions are imprecise.

Five states per instruction: **IF → ID → IS → EX → WB**. The recommended loop calls stages **in reverse order**:

```
do { Retire(); Execute(); Issue(); Dispatch(); Fetch(); } while (Advance_Cycle());
```

Calling stages back to front within a cycle means each stage sees the space freed by the downstream stage in
the same cycle, while an instruction advances at most one stage per cycle. It is the simulator equivalent of
pipeline registers clocking simultaneously. Calling them front to back would let an instruction fetched this
cycle dispatch in the same cycle.

Timing semantics (what each printed `{begin, duration}` means):
- **IF:** always 1 cycle, because every fetched instruction becomes ID on the next cycle's Dispatch() call.
- **ID:** the cycles spent waiting for scheduling-queue space and dispatch bandwidth, in program order.
- **IS:** the cycles waiting for operands (wakeup) and issue bandwidth (oldest ready first).
- **EX:** exactly the latency (1, 2 or 10).
- **WB:** 1 cycle.

Renaming on dispatch: a source is ready if the register-file entry is ready; otherwise it copies the
producer's tag. The destination sets `reg.tag = my_tag; reg.ready = false`. Sources are renamed **before**
the destination (handles `r1 = r1 + r2`). When an instruction completes, it wakes up queue entries with
matching source tags, and it marks the register ready **only if `reg.tag == my_tag`**.

Worked example (first lines of `gcc_val.txt`, N=8, S=64):
```
0 fu{0} src{29,14} dst{-1} IF{0,1} ID{1,1} IS{2,1} EX{3,1}  WB{4,1}
1 fu{2} src{29,-1} dst{14} IF{0,1} ID{1,1} IS{2,1} EX{3,10} WB{13,1}
4 fu{2} src{14,-1} dst{14} IF{0,1} ID{1,1} IS{2,11} EX{13,10} WB{23,1}
```
Instruction 4 reads r14, which instruction 1 produces (latency 10). It waits in IS from cycle 2 until 1's
broadcast at cycle 13, giving IS duration 11, and only then executes.

### 5.8 Project 3 results and how to interpret them

Here is the IPC from my (validated) simulator. Columns are the scheduling-queue size S; I added rows and columns
beyond the assignment for insight.

gcc:

| N \ S | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.95 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.51 | 1.82 | 1.96 | 1.99 | 1.99 | 1.99 | 1.99 | 1.99 |
| 4 | 1.81 | 2.64 | 3.46 | 3.83 | 3.94 | 3.94 | 3.94 | 3.94 |
| 8 | 1.85 | 2.93 | 4.54 | 6.11 | 7.39 | 7.66 | 7.66 | 7.66 |
| 16 | 1.85 | 2.95 | 4.73 | 7.48 | 10.85 | 14.33 | 14.77 | 14.77 |

perl:

| N \ S | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| 2 | 1.22 | 1.60 | 1.89 | 1.96 | 1.97 | 1.97 | 1.97 | 1.97 |
| 4 | 1.40 | 2.02 | 2.94 | 3.68 | 3.85 | 3.89 | 3.89 | 3.89 |
| 8 | 1.43 | 2.14 | 3.37 | 5.09 | 6.99 | 7.47 | 7.59 | 7.59 |

The instruction mix is gcc 70.5% type 0, 10.2% type 1, 19.3% type 2, and perl 65.9% / 9.2% / 24.9%.

What to say about it:
1. **IPC ≈ min(N, ILP exposed by a window of S).** At small S (8) IPC is about 1.85 for every N ≥ 4.
   The window, not the width, is the bottleneck (*window-limited regime*). At large S, IPC → N
   (*width-limited regime*).
2. **The saturation knee in S grows with N** (gcc: about 32–64 for N=2, about 128 for N=4, about 256 for
   N=8, about 512 for N=16). This is Little's law. Sustaining N instructions per cycle when each spends W
   cycles in the queue (waiting for a 10-cycle producer chain) needs about N × W entries.
3. **perl needs a bigger window than gcc** for the same N (at N=8, S=128: 6.99 against 7.39) because 25% of
   its instructions are 10-cycle operations, which lengthens W.
4. **Interaction:** raising N without raising S buys almost nothing (N=4 → 8 at S=16: 2.64 → 2.93).
   Raising S without raising N is capped at N. Width and window must scale together, which is why real cores
   grow ROB, issue-queue and register-file sizes along with width.
5. **Why IPC is so high** (14.8 at N=16) compared with real cores: perfect caches, perfect branch prediction,
   no structural limits beyond width, and no ports or register-file read limits. The model isolates
   dynamic scheduling. Saying this explicitly shows understanding.
6. Things that don't apply to this model, and that my report wrongly invoked: cache misses, memory stalls,
   ALU/FPU contention. The FUs are universal and fully pipelined. N is superscalar **width**, not a thread
   count, yet my plot legend called it "Number of Threads".

---

## 6. Thread-level parallelism (overview; CS6400 covers it in depth)

- **Cache coherence:** the single-writer/multiple-reader (SWMR) invariant plus the data-value invariant.
  - **Snooping protocols** (bus or broadcast): MSI → MESI (the E state avoids an upgrade for private data)
    → MOESI (the O state supplies dirty data without a writeback). Write-invalidate versus write-update.
  - **Directory protocols** for scale: a home node tracks sharers (full bit-vector, limited pointers or
    coarse vectors). Transactions take 3 hops (requester → home → owner → requester) or 4 hops.
  - **False sharing:** independent variables in one line ping-pong between cores. Fix it with padding or
    alignment to 64 B.
- **Memory consistency:** sequential consistency (SC; Lamport 1979); **TSO** (x86: store buffer, so a load
  may overtake an earlier store to a different address); relaxed models (ARM, RISC-V RVWMO, Power) need
  **fences** or acquire/release. Language models (C++11) sit above the hardware model.
- **Synchronisation:** test-and-set, test-and-test-and-set, CAS, LL/SC, fetch-and-add; locks, barriers;
  lock-free data structures and the ABA problem.
- **Multithreading:** coarse-grained (switch on a long stall), fine-grained (switch every cycle; GPUs are
  extreme fine-grained MT), and **SMT** (issue from several threads in one cycle; Intel Hyper-Threading).

## 7. Data-level parallelism (overview; CS6023 covers GPUs in depth)

- **Vector architectures:** vector registers, vector length register (VLR), strip-mining, chaining (forwarding
  between vector units), multiple lanes, gather/scatter, mask registers. Cray-1 is the archetype; RISC-V V is
  the modern revival.
- **SIMD extensions:** SSE, AVX2, AVX-512, NEON, SVE (length-agnostic).
- **GPUs:** SIMT, warps, massive multithreading for latency hiding, and a memory hierarchy of shared
  memory/L1, L2 and HBM. Tensor cores perform matrix-multiply-accumulate.
- **Roofline model** (Williams, Waterman & Patterson 2009):
  `attainable FLOP/s = min(peak FLOP/s, arithmetic intensity × memory bandwidth)`.
  Indispensable for reasoning about ML kernels.

---

## 8. My projects: specification, what I built, verification, defects, fix list

The three CS6600 project directories (`Project_1`, `Project_2`, `Project_3`) each contain the original zip, the
spec PDF, traces, reference outputs, a Dockerfile (Ubuntu 22.04 + g++), my source code and, for Projects 2 and 3,
my report. The grading pattern is roughly 30–40% for a simulator that passes the validation and mystery runs
(TAs diff the output with `diff -iw`), with the rest for the report.

### 8.1 Project 1: Cache and memory hierarchy simulator (due 27 Sep 2024)

**Spec summary:** a generic `CACHE` class instantiable as L1 or L2, with an optional fully associative LRU
victim cache inside L1; WBWA; LRU; non-inclusive L2. The CLI is
`./cache_sim L1_SIZE L1_ASSOC L1_BLOCKSIZE VC_NUM_BLOCKS L2_SIZE L2_ASSOC trace_file`.
It must print the configuration, the final contents of L1, the VC and L2 (MRU→LRU, `D` for dirty), 16 raw
statistics (a–p), AAT, EDP and area, all matching the reference outputs. The report needs 7 plots plus
discussion (Section 2.11).

**Files in `Project_1/`:** `Shubh_CacheSimulator.cpp` (latest, 1 Oct 2024), `cache_sim.cpp`, `cache.c`,
`temp.cpp`, `C_Sim.cpp`, `CacheSimulator.cpp` (earlier attempts), and `outputf.txt` (a run of the latest
version). No report exists in the folder.

**Verification I performed:** I compiled the latest version and ran it on the four L1 1 KB 2-way validation
configurations.

| Config | Result |
|---|---|
| no VC, no L2 | crash: SIGFPE (exit 136) |
| VC 16, no L2 | crash: SIGFPE |
| no VC, L2 8 KB | crash: SIGFPE |
| VC 16 + L2 8 KB | runs, but stats are wrong (e.g. 0 swap requests vs 15,938; L1 read misses 15,885 vs 8,322; L2 writes 0 vs 7,598; AAT/EDP/area all 0) |

`cache_sim.cpp`, the filename the provided Makefile expects, **does not compile**, because a data member named
`CACHE` shadows the class name. Under the TA workflow (`make`, then `./cache_sim <args>`) nothing builds or runs.

**Defects in `Shubh_CacheSimulator.cpp`** (line numbers refer to that file):
1. **CLI:** `main()` reads 8 tokens from `stdin` (`cin >> command >> ...`, l.395–401) instead of `argv`.
2. **Division by zero:** the constructor computes `size / (assoc * blockSize)` (l.74). With `VC_NUM_BLOCKS=0`
   or `L2_SIZE=L2_ASSOC=0` that is 0/0, which crashes 3 of the 4 validation shapes. The objects for absent
   caches shouldn't be constructed at all.
3. **Output goes to `outputf.txt`** (l.463–478) instead of stdout, so a `diff` of the console output fails.
4. **Shared global counters:** L2 is accessed through the same `Cache_Access()` that increments the *L1* miss
   counters (l.118–125), so L1 misses are inflated by L2 misses. On an L2 miss the block is allocated twice
   (inside `Cache_Access`, then again at l.144).
5. **Writebacks never reach L2:** `Block_Allocation()` only increments a counter for a dirty victim
   (l.184–187); it never issues a write to the next level. L2 writes and L2 writebacks are always 0, and every
   L2 eviction is miscounted as an L1/VC writeback.
6. **Victim cache logic is wrong throughout:**
   - The VC stores L1 *tags* (which lose the index bits) but is searched with the full 32-bit binary address
     string (l.201), so it essentially never hits.
   - `swapRequest` is only incremented on a successful swap; it should count every VC probe made with a full
     set.
   - On a VC miss the L1 victim is never inserted into the VC.
   - VC replacement evicts `cache[0].back()` (a fixed slot), not LRU, and VC LRU state is never updated.
   - After a swap the block's dirty bit is discarded (`dirty_Bit = false`) and the write isn't applied.
7. **No CACTI integration:** `parse.h`'s `get_cacti_results()` is never called, so AAT, EDP and area print 0.
8. **Format:** labels and section names differ, numbers aren't printed with 4 decimals, L2 is printed even
   when absent, invalid ways print `Empty`, and VC contents are never printed.
9. **Design:** addresses as strings (`hextobinary`, `substr`, `stoi` on every access) are slow and error-prone.
   Use `uint32_t` with shifts and masks.

**Phase-2 fix plan:** rewrite from scratch around one `Cache` class: integer addresses; per-set
`std::vector<Block>` ordered MRU→LRU (or LRU counters); a `next` pointer (nullptr means memory); an optional
VC owned by L1; the allocation order "victim writeback → read X". Integrate CACTI (with the 0.2 ns fallback).
Reproduce the exact reference format. Pass all 8 validation runs with `diff -iw`. Script every experiment
(Plots 1–7) and write a data-driven report. (The `cacti` binary is x86-64 and runs on this machine.)

### 8.2 Project 2: Branch predictor simulator (due 20 Oct 2024)

**What I built:** `branch_predictor.cpp`, an abstract `BranchPredictor` base class with `BimodalPredictor`
and `GsharePredictor` subclasses. It parses argv correctly, the index and GHR math are exactly right
(Section 4.5), and it prints the table contents.

**Verification:** I compiled and diffed against all 6 validation runs plus 2 sample runs. Every number and
every table entry matches. **However, a strict `diff -iw` fails all 8**, because `report()` prints one extra
blank line after the misprediction rate (`std::cout << std::endl;`), and `-w` does not ignore blank lines
(that would need `-B`). The grader's script would have flagged every run as a mismatch. It is a one-line fix.

**Report issues:**
- Budget arithmetic error: m ≤ 13 instead of **m ≤ 16** (Section 4.6), so the space within budget was never
  explored.
- The discussion is generic and hedged ("may show", "might show", "assume it plateaus at m = 10") and doesn't
  read the actual data. It claims "smaller n could lead to higher misprediction", which the gcc data
  contradicts: n = 2 is the best small-table gshare for gcc.
- The proposed designs are poor (gcc gshare (10,4) at 17.99%; jpeg gshare (9,6) at 8.13%). The correct
  analysis is in Section 4.7.

**Phase-2 fix:** drop the blank line, sweep m = 7…16 for all n, regenerate the plots (one curve per n, legal m
only), and rewrite the discussion around aliasing versus correlation, the diminishing-returns knee and a cost
table.

### 8.3 Project 3: Dynamic scheduling (Tomasulo) simulator (due 16 Nov 2024 after the spec update)

**What I built:** `proj_tomasulo.cpp`, following the recommended structure: the five states, `aROB`
(bookkeeping), the dispatch list, scheduling queue and execute list, the reverse-order stage calls, a
128-entry register file `{tag, ready}`, oldest-first issue, and wakeup by tag broadcast.

**Verification:** both validation runs (`gcc` N=8 S=64; `perl` N=3 S=16) pass `diff -iw` exactly.

Minor notes:
- `Retire_Instruction()` is a no-op (it `break`s immediately). That doesn't affect correctness, but the ROB
  grows unboundedly and `AdvanceCycle()` scans it every cycle (O(n²); fine for 10K instructions).
- `stoi(strPC, nullptr, 16)` would throw for PCs ≥ 0x80000000. Use `stoul`.
- There are several stale copies (`OriginalFile.cpp`, `ooosim_OOP.cpp`, `ooosim.cpp`).

**Report issues:** it's generic, quoting no numbers. It attributes saturation to "cache misses", "memory
stalls" and "ALU/FPU contention", none of which exist in this model. It claims IPC "may decline" with larger
S. The legend says "Number of Threads (N)", but N is superscalar width. The x-axis is linear for
power-of-two S (log2 would be cleaner). Section 5.8 is the analysis it should contain.

**Phase-2 fix:** replace the discussion with the data-driven analysis above, fix the legend and axis, and
optionally add N=1 and N=16 curves and a Little's-law argument.

---

## 9. Revision questions with answers

1. **Why does a direct-mapped cache sometimes beat a 2-way cache in AMAT despite a higher miss rate?**
   Its hit time is lower (no way-select mux, no parallel tag compare), and every access pays the hit time.
   If MR × MP gains less than the hit-time difference, DM wins.
2. **Local vs global miss rate of L2?** Local = L2 misses / L2 accesses (L1 has filtered out the easy
   accesses, so it looks high). Global = L2 misses / CPU accesses = MR_L1 × MR_L2,local.
3. **When does a victim cache help most?** For direct-mapped or low-associativity caches with ping-ponging
   conflict misses. It doesn't help with capacity or compulsory misses.
4. **Why skip the VC search when the set has an invalid way?** A block reaches the VC only by eviction from a
   full set. A non-full set has never evicted anything, so the VC can't hold a block of that set.
5. **Why does gshare underperform bimodal with small tables?** History multiplies the number of table entries
   each static branch uses. In small tables that raises aliasing and training time more than correlation
   gains.
6. **How does Tomasulo remove WAR/WAW hazards?** Sources capture values or producer tags at dispatch; the
   destination maps to the new producer's tag. Later writers never overwrite what an earlier reader needs,
   and only the latest producer updates the architectural register.
7. **Why call pipeline stages in reverse order in a cycle-level simulator?** So that each stage sees the
   resources freed downstream in the same cycle while instructions advance at most one stage per cycle. It
   mimics pipeline latches.
8. **What determines the ROB/IQ size needed to sustain a given IPC?** Little's law: entries ≈ IPC × average
   residence time (latency + dependence waits + commit delay).
9. **What is a precise exception and how does a ROB provide it?** All instructions before the faulting one have
   committed and none after have modified architectural state. The ROB holds results until in-order commit
   and flushes younger entries.
10. **What limits the clock rate and the IPC of deeper and wider pipelines?** Latch overhead and clock skew,
    the branch misprediction penalty, wakeup/select loop latency, register-file ports and the bypass network.
11. **Inclusive vs exclusive LLC?** Inclusive simplifies snoop filtering but wastes capacity and needs
    back-invalidation. Exclusive maximises capacity but adds data movement.
12. **Why did Dennard scaling's end push the industry to accelerators?** Power density became the limit, and
    specialisation gives more performance per watt than general-purpose ILP.

---

## 10. Bridges from CS6600 to AI infrastructure

These are the places where this course's mental models map directly onto today's LLM systems. They're useful
both for interviews and for picking resume projects.

| CS6600 concept | AI-infrastructure counterpart |
|---|---|
| Caches, LRU, inclusion, victim caches | **KV-cache management** in LLM serving: vLLM's **PagedAttention** (Kwon et al., SOSP 2023) applies virtual-memory paging to the KV cache; **prefix caching** (SGLang RadixAttention) is a cache with an eviction policy; tiered KV offload (GPU HBM → CPU DRAM → SSD) is a multi-level hierarchy with victim-like tiers. |
| AMAT, miss penalty, bandwidth-aware miss cost | **Decode is memory-bandwidth bound:** time per token ≈ bytes of weights + KV read / HBM bandwidth. Quantisation and batching change "accesses per useful FLOP". This is roofline thinking. |
| EDP and energy accounting | **Energy per token** and TCO metrics for inference clusters; power capping. |
| Branch prediction (predict, verify, roll back) | **Speculative decoding** (Leviathan et al., ICML 2023; Chen et al. 2023): a draft model is the "predictor", the target model verifies, the acceptance rate is prediction accuracy and rejection is a rollback. History-based predictors correspond to n-gram / lookahead / Medusa / EAGLE drafters. Also **MoE expert-routing prediction** for prefetching experts. |
| Tomasulo / dataflow scheduling, wakeup/select | Asynchronous **dataflow execution of operator graphs** (CUDA Graphs, streams/events, XLA/TVM schedulers); **pipeline-parallel training schedules** (GPipe, 1F1B; bubbles are structural hazards); dependency tracking in task runtimes (Ray, Legion). |
| Little's law (window size = throughput × latency) | **Continuous batching** in LLM serving (Orca, OSDI 2022): concurrency = throughput × latency; GPU occupancy for latency hiding; KV-cache capacity limits concurrency the way ROB size limits ILP. |
| Trace-driven architectural simulation, validated by diffing | **Performance simulators for LLM systems**: Vidur (MLSys 2024) for serving, ASTRA-sim for distributed training, LLMCompass (ISCA 2024) for hardware design-space exploration. Building or extending a validated LLM-inference simulator is a natural, distinctive project for someone with this background. |

---

## 11. References

Textbooks
- J. L. Hennessy, D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed. (Morgan Kaufmann,
  2017). Ch. 1 (quantitative principles), Ch. 2 + App. B (memory hierarchy), Ch. 3 + App. C (ILP, pipelining),
  Ch. 4 (DLP), Ch. 5 (TLP).
- J. P. Shen, M. H. Lipasti, *Modern Processor Design: Fundamentals of Superscalar Processors* (2005).
- D. J. Sorin, M. D. Hill, D. A. Wood, *A Primer on Memory Consistency and Cache Coherence*, 2nd ed. (2020).

Papers
- N. P. Jouppi, "Improving Direct-Mapped Cache Performance by the Addition of a Small Fully-Associative Cache and
  Prefetch Buffers", ISCA 1990.
- M. D. Hill, A. J. Smith, "Evaluating Associativity in CPU Caches", IEEE Trans. Computers 1989 (3C model).
- S. Wilton, N. Jouppi, "CACTI: An Enhanced Cache Access and Cycle Time Model", IEEE JSSC 1996; N. Muralimanohar
  et al., "CACTI 6.0", HP Labs TR 2009.
- J. E. Smith, "A Study of Branch Prediction Strategies", ISCA 1981.
- T.-Y. Yeh, Y. N. Patt, "Two-Level Adaptive Training Branch Prediction", MICRO 1991.
- S. McFarling, "Combining Branch Predictors", DEC WRL Technical Note TN-36, 1993 (gshare, tournament).
- D. A. Jiménez, C. Lin, "Dynamic Branch Prediction with Perceptrons", HPCA 2001.
- A. Seznec, P. Michaud, "A Case for (Partially) TAgged GEometric History Length Branch Prediction", JILP 2006.
- R. M. Tomasulo, "An Efficient Algorithm for Exploiting Multiple Arithmetic Units", IBM J. R&D, 1967.
- J. E. Smith, A. R. Pleszkun, "Implementing Precise Interrupts in Pipelined Processors", ISCA 1985 / IEEE TC 1988.
- K. C. Yeager, "The MIPS R10000 Superscalar Microprocessor", IEEE Micro 1996.
- D. W. Wall, "Limits of Instruction-Level Parallelism", ASPLOS 1991.
- G. Z. Chrysos, J. S. Emer, "Memory Dependence Prediction Using Store Sets", ISCA 1998.
- A. Jaleel et al., "High Performance Cache Replacement Using Re-Reference Interval Prediction (RRIP)", ISCA 2010.
- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model", CACM 2009.

AI-systems bridges
- W. Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention", SOSP 2023.
- G.-I. Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022.
- Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
- A. Agrawal et al., "Vidur: A Large-Scale Simulation Framework for LLM Inference", MLSys 2024.
- S. Rashidi et al., "ASTRA-SIM: Enabling SW/HW Co-Design Exploration for Distributed DL Training", ISPASS 2020.
- H. Zhang et al., "LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference", ISCA 2024.
