# CS6570 — Secure Systems Engineering (IIT Madras, Jan–May 2025/2026) — Learning Notes

> Instructor: **Prof. Chester Rebeiro** (CSE, IIT Madras; CASL lab). H slot, room CS24.
> Course page: `sites.google.com/cse.iitm.ac.in/cs6570-2025/`. The practice material is the NPTEL
> "Secure Systems Engineering" course (same instructor), bundled under `Practice/casl-sse-*`.
> This single file holds: lesson-by-lesson notes for the whole syllabus (binary exploitation, defences, access
> control, confinement, TEEs, hardware security, microarchitectural attacks); a review of my four assignments
> (buffer overflow, ROP, format string, heap), each with the stack/heap reasoning and the marks rationale; my
> course project on AI-agent security (indirect prompt injection); revision Q&A; and bridges to AI infrastructure.
>
> **Evaluation:** Assignments 40, Quiz 1 20, Quiz 2 20, End-sem / project 20.
> **Structure (from L1):** Part 1 Binary Exploitation → Part 2 Malware Analysis → Part 3 Designing a Secure
> System. Deep dive into software (assembly, compiler, OS level), analysis techniques (static/dynamic/symbolic,
> statistical + ML), a reading assignment and a CTF.
>
> *Note on scope.* The assignments are authorised exploitation exercises against instructor-provided binaries in a
> sandboxed Docker VM with ASLR/PIE/canary deliberately disabled. These notes document the **mechanisms, stack/heap
> layouts and offsets** for revision; they do not reproduce full raw payload byte strings, which live in my
> submitted `payloadCreate*.py` / `exploit_stage*.py` scripts.

---

## TL;DR — status of my work

| Item | What it is | Verdict |
|---|---|---|
| A1 (buffer overflow) | Bypass a custom stack canary, call `exploit()`, clean exit | **Complete & correct.** Offsets verified: buf at ebp−0x10, canary +4, ret +20; 28-byte payload |
| A2 (return-to-libc / ROP) | Call `winners()` with password; ROP write-what-where to set `id=100` | **Complete & correct.** 5-gadget chain sources the null-byte value `0x64` from inside the binary; all addresses shell-safe |
| A3 (format string) | `%s` read of `secret`; `%hn` GOT overwrite of `putchar` → `flag()` | **Complete & correct.** Primary (pure format string) + documented alternative (FSB + overflow) |
| A4 (heap, glibc 2.23) | S1 UAF fastbin leak; S2 poison-null-byte + unlink; S3 BSS leak | **Complete.** All three stages solved. S3 leaks `session_pin` via the `vault_audit` read primitive rather than the "overwrite" path the spec hints at — still passes the auto-grader (`value == session_pin`) |
| Project | Indirect prompt injection on an agentic workflow; two mitigations; an LLM security gateway proposal | **Strong, directly relevant to Phase 3** |

---

## Table of contents

1. Course overview and the security mindset
2. Program binaries: ELF, processes, the stack
3. Binary exploitation 1 — buffer overflows and shellcode
4. Defences — canaries, W^X, ASLR/PIE (GOT/PLT), bounds checking (SoftBound)
5. Binary exploitation 2 — return-to-libc and ROP
6. More vulnerabilities — buffer overreads, format strings, integer overflows
7. Heap exploitation — ptmalloc2 internals and attacks
8. Access control, information flow, confinement, SFI
9. Trusted execution environments (TrustZone, SGX)
10. Hardware security — PUFs, hardware Trojans, power analysis
11. Microarchitectural attacks — cache timing, Meltdown, Spectre, Rowhammer
12. My assignments (A1–A4): analysis, verification, marks rationale
13. My project: AI-agent security / indirect prompt injection
14. Revision Q&A
15. Bridges to AI infrastructure
16. References

---

## 1. Course overview and the security mindset

- **Where did we go wrong?** Mainframes (1960–75, ~$1M, cold-war era) treated security as the prime concern. The
  PC/VLSI generation (70s) optimised for ease of development (C/C++, Unix/Windows, monolithic, plug-and-play) and
  made security "less of a concern". Then the internet, IoT and cloud connected everything built on that insecure
  base. C/C++ is now a legacy we cannot rewrite.
- **The attacker needs one flaw, any flaw.** Flaws come from design, the human factor, or bugs. Classes: SQL
  injection (application), buffer overflow/overread, heap (double-free, use-after-free), integer overflow, format
  string (system software), peripherals (USB, printers), hardware Trojans, and covert channels.
- **Defence in depth (the course's spine):**
  1. **Eliminate** flaws — safer languages/libraries, hardware help, static analysis.
  2. If you can't eliminate, **make subversion harder** — W^X, ASLR, canaries.
  3. If malware still runs, **detect it at runtime** — signatures, ANN/ML.
  4. If you can't detect it, **restrict what it can do** — sandbox, access control, virtualization, TrustZone, SGX,
     and information-flow tracking (DIFT).
- **Parts of malware:** (a) **subvert execution** — change the normal control flow; (b) **payload** — the code the
  attacker wants to run. Covert channels don't subvert execution but still leak confidentiality.

---

## 2. Program binaries: ELF, processes, the stack

- **ELF file** (`gcc hello.c`): ELF header → program header table → sections → section header table.
  - **Object (`.o`, linker view):** organised in **sections** (`.text`, `.data`, symbol table, relocation info).
    `readelf -h` shows identification, type (relocatable / executable / shared / core), machine, entry point,
    pointers to program/section headers. Section types: PROGBITS, SYMTAB, NULL, NOBITS (occupies no file bytes,
    e.g. `.bss`), RELA.
  - **Executable (loader view):** organised in **segments** (program headers). Each header: type, file offset,
    vaddr, paddr, file size, mem size, flags (R/W/X). `readelf -l` shows the segment↔section mapping.
  - `objdump --disassemble-all` dumps the code.
- **Process creation:** `fork()` clones (returns child PID to parent, 0 to child); `execlp()` replaces the image.
- **Process virtual memory map** (low→high): text (instructions) → data → heap (grows up) → … → stack (grows down),
  up to MAX. `cat /proc/<pid>/maps` shows ranges, flags, and backing file.
- **Stack frame (x86, 32-bit):**
  - `%ebp` = frame pointer, `%esp` = stack pointer.
  - Caller pushes args (right to left), then `call` pushes the return address.
  - Callee prologue: `push %ebp; mov %esp, %ebp; sub $N, %esp` (allocates locals).
  - Layout from the frame: `[ebp+8..]` args, `[ebp+4]` return address, `[ebp]` saved ebp, `[ebp-4..]` saved regs
    and locals.
  - `leave` = `mov %ebp,%esp; pop %ebp`; `ret` pops the return address into EIP.

---

## 3. Binary exploitation 1 — buffer overflows

- **The bug.** `strcpy(buf, input)` into a fixed `char buf[N]` with no bound check. Writing past `buf` overwrites,
  in order, saved registers, saved ebp, and the **saved return address**.
- **Skip an instruction** → overwrite the return address with an address further along.
- **Hijack control** → overwrite the return address with the address of attacker code.
- **Classic stack-smashing exploit (Aleph One):**
  1. Assemble shellcode (e.g. `execve("/bin/sh")`). `objdump` the machine bytes.
  2. **Remove 0x00 bytes** — `strcpy` stops at NUL. Rewrite instructions to avoid them (e.g. `xor` to zero a
     register instead of `mov $0`).
  3. Find a stack buffer overflow.
  4. Build a large string: a **NOP sled** (`0x90…`), the shellcode, then many copies of the **buffer address (BA)**
     so that one of them lands on the return slot. The NOP sled widens the margin for an imprecise BA.
  5. On `strcpy` return, EIP jumps into the sled and slides into the shellcode → `$ sh`.
- **In the wild:** **Code Red** (July 2001) overflowed a buffer in Microsoft IIS via a `GET /default.ida?NNNN…`
  request; 359,000 machines in six days.
- `objdump` offset arithmetic: a declared `char buf[12]` usually allocates `0x14` (20) bytes for alignment; the
  offset from buffer start to the return slot is what the exploit needs, found by GDB/GEF (`x/20wx $esp`) or trial.

---

## 4. Defences

### 4.1 Stack canaries

- A known (ideally random) value placed between the locals and the saved return address; checked in the epilogue.
  A buffer overflow that reaches the return address also clobbers the canary → "stack smashing detected" → abort.
- gcc: off by default historically; `-fstack-protector` adds them by heuristic (buffer ≥ 8 bytes, use of
  `strcpy`/`scanf`, …). Stored via the `%gs` thread-local segment.
- **Weaknesses:** a canary can be left intact if you don't overwrite it (e.g. a write primitive that jumps over
  it), and a **non-random / hardcoded** canary is trivially replayed (**exactly my A1**). Production canaries are
  random per-run and include a NUL byte to stop string-based reads.

### 4.2 W^X / NX / DEP

- Intel/AMD NX bit marks non-code pages non-executable; executing there faults. In Linux since 2004, Windows as DEP.
- Breaks shellcode-on-stack. Does **not** help JIT compilers that must execute generated code (they disable it).
- Does not stop **return-to-libc / ROP**, which reuse existing executable code (→ §5).

### 4.3 ASLR and position independence

- **ASLR** randomises base, libraries, heap and stack per execution, so a hardcoded exploit address fails. Linux
  PaX, 2001; now default. `/proc/sys/kernel/randomize_va_space`: 0 off; 1 stack/VDSO/mmap; 2 (default) also the
  data segment. Set permanently via `/etc/sysctl.conf` + `sysctl -p`.
- **Making code relocatable:**
  - **Load-time relocation**: the loader patches addresses — slow load, needs a writable code segment, no code
    sharing.
  - **PIE/PIC**: code runs at any address via relative addressing and a **GOT (Global Offset Table)**.
- **GOT**: a table at a fixed location (per module) holding the absolute addresses of external globals/functions.
  Built once at load time, lives in the writable data segment (so pages aren't shared anyway). A global access
  becomes: get EIP (`__x86.get_pc_thunk`), add the GOT offset, load the real address from the GOT.
- **PLT (Procedure Linkage Table) + lazy binding**: function calls go through `func@plt`. On the first call PLT[n]
  jumps to GOT[n], which jumps back into PLT[n]'s resolver stub; the resolver finds `func`, writes its address into
  GOT[n], and calls it. Subsequent calls jump straight through the now-filled GOT[n]. This defers symbol resolution
  until first use.
- **Bypassing ASLR:** brute force (32-bit has low entropy), **return-to-PLT**, **overwriting the GOT** (→ my A3),
  info leaks (format string, overread), and timing attacks. A **writable GOT (partial RELRO)** is what makes GOT
  overwrite possible; full RELRO + BIND_NOW makes the GOT read-only after load.

### 4.4 Safer languages and compiler bounds checking

- Memory-safe languages (Java, C#) are slow / unfit for systems code; **Rust** is the promising replacement.
  C11 Annex K adds `_s` functions (`gets_s`, `strcpy_s`, `strncpy_s`).
- **SoftBound** (Nagarakatte et al.): associate a **base and bound** with every pointer; check before each
  dereference; checks inserted at compile time; metadata in a disjoint table. Pointer arithmetic inherits base/bound
  (checked only at dereference). Pointers passed in registers need the function signature rewritten to carry
  base/bound. (This is the dynamic cousin of CS6843's static range/polyhedral bounds checking.)

---

## 5. Binary exploitation 2 — return-to-libc and ROP

### 5.1 Return-to-libc

- Defeats NX by returning into **existing** executable code (libc) instead of stack shellcode.
- Overwrite the return address with the address of a libc function (e.g. `system`), set up a fake frame so that its
  argument (a pointer to `"/bin/sh"`) and its own return address (e.g. `exit`, for a clean exit) are in place.
  `"/bin/sh"` is found among the process's environment strings near the bottom of the stack.
- **Limitation:** restricted to whole library functions, which can be removed from the library.

### 5.2 Return-oriented programming (ROP)

- **Gadget**: a short instruction sequence ending in `ret` (opcode `0xc3`). Useful gadgets don't transfer control
  out of themselves except via the final `ret`.
- **Build the payload as a chain of gadget addresses on the stack.** `ret` pops the next gadget address into EIP
  and advances ESP, so the stack becomes an instruction stream. `ret` ≈ "NOP that advances the program counter" in
  ROP's execution model.
- **Finding gadgets:** scan the binary for `0xc3`, then walk backwards building a trie; each root-to-node path is a
  candidate gadget. **Intended vs unintended** instructions: x86 is variable-length and dense, so decoding
  mid-instruction yields many *unintended* gadgets (`F7 C7 07 00 00 00 0F 95 45 C3` reinterpreted). Shacham found
  15,121 nodes in ~1 MB of libc. RISC (fixed-width, aligned) has low "geometry" → far fewer gadgets. Tools:
  **ROPgadget**, **Ropper**.
- **ROP is Turing-complete on x86.** Primitives:
  - load constant (`pop %edx; ret`);
  - load from memory (`mov 64(%edx), %eax`);
  - store (`mov %eax, 24(%edx)`);
  - add (`addl (%edx), %eax`);
  - unconditional branch = `pop %esp` (change the stack pointer);
  - conditional branch: set flags (`cmp`/`neg`), transfer flags to memory (`lahf`/`pushf`, or indirectly via `adc`/
    `rcl`), then perturb ESP by a flag-dependent offset.
- **My A2 is a textbook write-what-where ROP** (details in §12.2).
- **Defences:** full RELRO, ASLR + PIE, CFI (control-flow integrity), shadow stacks / Intel CET, coarse-grained ROP
  detectors (kBouncer).

---

## 6. More vulnerabilities

### 6.1 Buffer overreads

- Reading past a buffer leaks adjacent memory. **Canaries, W^X and ASLR do not stop overreads** (no write, no code
  execution, no segment escape).
- **Heartbleed** (CVE-2014-0160, OpenSSL TLS heartbeat): the response is built from an attacker-controlled
  `payload_length` (up to 65535) that was never checked against the actual record length `D_length`. The server
  `memcpy`s `payload_length` bytes starting at a 1-byte payload, returning up to ~64 KB of its own heap (keys,
  sessions) per request. Patch: discard if `payload_length > D_length`.

### 6.2 Format string vulnerabilities

- **The bug:** `printf(user_string)` instead of `printf("%s", user_string)`. `printf` has no way to know the real
  argument count — it pulls one stack slot per specifier.
- **Primitives:**
  - crash: `%s%s%s%s…` (dereferences garbage);
  - dump the stack: `%x %x %x %x`;
  - **read any address**: place the target address in the string, use `%N$s` (direct parameter access) to jump to
    the Nth stack slot — which holds your address — and `%s` prints from it until NUL;
  - **write any address**: `%n` writes the number of characters printed so far to the pointed-to location. Pad with
    field widths to control the count. `%hn` writes 16 bits (write a 32-bit value as two 16-bit halves, so the
    character count stays small). Targets: return addresses, function pointers, **GOT entries**.
- **My A3** reads `secret` with `%s` and overwrites `putchar@GOT` with `%hn` so the next `putchar` jumps to `flag()`
  (details in §12.3).

### 6.3 Integer overflows

- **Width overflow:** storing a wide value in a narrow type (`(char)0x11223344 = 0x44`, `(short)… = 0x3344`); an
  `int→short` cast can make a size check pass yet overflow a later buffer.
- **Arithmetic overflow:** `malloc(len * sizeof(int))` with a `len` chosen so the product wraps → under-sized
  allocation → heap overflow.
- **Sign confusion:** a negative signed `len` passes `if (len > MAX)` but is reinterpreted as a huge unsigned value
  by `memcpy`'s `size_t` → massive copy/overread. `table + pos` with negative `pos` writes before `table`.
- **Stagefright** (Android, 2015): MP4 parsing, `uint8_t *buffer = new uint8_t[size + chunk_size]` where
  `chunk_size` comes from the file. On 32-bit `new` takes a 32-bit length (width overflow); on 64-bit `size +
  chunk_size` can wrap (arithmetic overflow) → heap exploit from a crafted MP4/MP3 message; ~1 billion devices.

---

## 7. Heap exploitation (glibc ptmalloc2)

### 7.1 Internals

- **Heap** = pool for dynamic allocation; manual (`malloc`/`free`), slower than the stack, for large/persistent
  objects. glibc uses **ptmalloc2** (derived from Doug Lea's dlmalloc).
- **Arenas.** A main arena (via `brk`) and per-thread arenas (via `mmap`, 132 KB each). Max arenas: 2× cores
  (32-bit) / 8× cores (64-bit). One arena serves one `malloc`/`free` at a time; too few arenas → contention.
- **Chunk** (`malloc_chunk`):
  - allocated: `prev_size | size` header (low bits are flags: **P/PREV_INUSE**, **M** mmapped, **N** non-main
    arena), then user data. `malloc(n)` user size ≈ `8 + (n/8)*8`; total chunk `N+8`. `mem` (returned) points past
    the header.
  - free: header + **fd/bk** pointers (forward/back in a bin list), reusing the data area.
  - If `P = 0`, the word before the chunk is the previous chunk's size (for backward coalescing).
- **Bins (free lists):**
  - **Fastbins**: ~10–12 single-linked LIFO lists for small sizes (16, 24, …, 80/128 B); **no coalescing** (fast
    but fragments).
  - **Unsorted bin**: a single list where freed chunks land first; `malloc` uses first-fit here.
  - **Small bins**: 62 bins < 512 B, circular doubly-linked, FIFO, with coalescing.
  - **Large bins**: 63 bins, sizes grouped (first 32 are 64 B apart, etc.), sorted by decreasing size within a bin.
  - **Top chunk**: the unused wilderness; split for requests, grown with `brk`/`sbrk`.
  - **Last-remainder chunk**: the leftover of the last split.
- **`free(p)`:** if the next chunk is allocated, mark this free (clear its P bit, set its size). If the previous or
  next chunk is free, **coalesce** (requires `unlink` from its bin).
- **`unlink(P)`** (remove a chunk from a doubly-linked bin): `FD = P->fd; BK = P->bk; FD->bk = BK; BK->fd = FD`.
  glibc 2.3.4+ adds the **safe-unlink** check `if (FD->bk != P || BK->fd != P) abort()` and a
  `size == next->prev_size` check. These catch *some* double-frees (the exact-adjacent case) but **most are not
  detected**.
- **First-fit reuse** (`how2heap/first_fit.c`): free `a`, allocate a smaller `c`, and `c` reuses `a`'s address — the
  basis of use-after-free.

### 7.2 Attacks

- **Use-after-free (UAF):** `free` leaves a dangling pointer the program still uses; a later allocation of the same
  size hands the attacker control of that memory. A freed fastbin chunk's first 8 bytes become the **fd pointer**,
  which leaks a heap address if the program prints freed data (**my A4 Stage 1**).
- **Unsafe unlink (classic Solar Designer):** overflow into a free chunk's fd/bk so that `unlink` performs
  `*(fd+0x18) = bk` and `*(bk+0x10) = fd` — a near-arbitrary write. The safe-unlink check is bypassed by pointing
  fd/bk at a known location (e.g. a global holding the chunk's own address, so `fd->bk == P`). **My A4 Stages 2–3**.
- **Poison null byte (off-by-one):** a single NUL written one byte past an allocation clears the next chunk's
  PREV_INUSE and shrinks its size, so a later `free` coalesces backward into an attacker-crafted fake chunk →
  **chunk overlap** (two live pointers to the same memory). **My A4 Stage 2**.
- **Metadata / fastbin corruption** (seen in the intro deck): double-free a fastbin chunk, then on the next `malloc`
  you get a chunk whose fd you control (`c`), write `GOT_entry − 12` into it, so a following `malloc` returns a
  chunk overlapping a GOT entry / function pointer → redirect a later call to the payload.
- Others: heap overflow, **heap spray**, House of Force/Spirit/Einherjar/Orange.

---

## 8. Access control, information flow, confinement, SFI

### 8.1 Access control

- **Who (subjects: users/processes) can do what (rights: r/w/x/share) to which objects (files, sockets, hardware).**
  Moving up the stack (hardware → OS → middleware → application) adds subjects, objects and complexity, lowering
  reliability.
- **Hardware mechanisms:** paging unit, privilege rings, interrupts (protect OS from apps, apps from each other,
  prevent hogging — confidentiality, integrity, availability).
- **Discretionary access control (DAC).** Access by requester identity; owners pass on rights.
  - **Access Matrix** (Lampson 1971): subjects × objects → rights; HRU formal model (commands, generic rights,
    primitive ops; the safety problem is undecidable in general).
  - Implementations: **ACL** (list stored with the object — "my name is on the list"; easy revocation, harder
    delegation) vs **capability** (an unforgeable ticket the subject holds — easy delegation, harder revocation).
- **Unix security:** subjects = users/groups/root; `setuid`/`setgid`; login via `crypt()` against `/etc/shadow`;
  `setuid`-root programs (e.g. `passwd` edits root-owned `/etc/shadow`); **file descriptor = capability** (once you
  hold it, no further checks); ptrace needs same uid. **Problems:** root is all-powerful (mitigated by BSD/OSX
  append-only/undeletable flags); uid permissions are coarse-grained (no "user X may run program Y to write file
  Z"; one owner + one group per file).
- **Information-flow policies** (because DAC ignores flow and is vulnerable to **Trojan horses** that inherit the
  user's privileges): assign every object a **security class** in a lattice; restrict flow (e.g. low→high only).
- **Mandatory access control (MAC) / multilevel security:**
  - **Bell–LaPadula (confidentiality, 1974):** **no read up** (simple security), **no write down** (★-property, so
    a Trojan at a high clearance can't leak down), plus the ds-property (every access also allowed by the matrix).
    MAC overrides DAC. Limits: ignores integrity, write-up corruption, and needs a rule for changing levels
    (**tranquility**: strong = labels never change; weak = labels don't change in a policy-violating way). Covert
    channels (page faults, locks, caches, branch predictors, timing) leak around it.
  - **Biba (integrity, 1975):** BLP upside down — **no read down**, **no write up**. (A general's order flows to
    all; a private's note can't influence the general.) In FreeBSD.

### 8.2 Confinement (running untrusted programs)

- Coarse options: air gap, VM, container. Finer: run each module as a separate **process** and talk over **RPC**;
  hardware isolates address spaces.
- A typical Apache web server runs everything in one process pool; a bug in any module/site ripples through all
  services and co-tenant sites (known Apache CVEs: data disclosure 2002, buffer overflow/RCE 2002, DoS 2003).
- **Principle of least privilege:** decompose into subsystems; grant minimal, fine-grained rights; narrow
  interfaces; assume exploits happen near the user (network edge); enforce from outside (OS).
- **OKWS** (Krohn, USENIX'04): each service a separate process, each in its own **chroot jail**, each a distinct
  unprivileged uid, narrow DB access. `okld` (root) bootstraps and relaunches; `okd` demuxes port 80 to services;
  `oklogd` owns logging in its own jail; `pubd` serves static files read-only. Unix tools: `chroot` (limit the file
  view), `setuid` (drop privilege), **passing file descriptors** (privileged parent opens, hands fd to unprivileged
  child). Strict 1-process-per-user confinement is strongest but slow; OKWS trades toward 1-process-per-service.

### 8.3 Software fault isolation (SFI)

- Goal: run untrusted C/C++ **inside one process** (e.g. a browser plugin, Google **NaCl**) without RPC overhead.
  Partition the address space into **fault domains**, each with a code segment and a data+stack+heap segment that
  share the same upper **segment-ID** address bits.
- Rewrite the untrusted module (compile- or load-time, with a verifier) so that:
  - **safe** instructions (most ALU ops, in-segment jumps/accesses resolvable statically) run as-is;
  - **unsafe** instructions (indirect jump/call `jmp *eax`, indirect load/store) are checked at runtime.
- Two runtime schemes (Wahbe et al., SOSP'93):
  - **Segment matching**: compute the target, compare its upper bits with the segment ID, trap on mismatch
    (detects + reports the faulting instruction; ~4 dedicated registers).
  - **Address sandboxing**: *force* the upper bits to the segment ID (`and eax, mask; or eax, segID`) — faster,
    can't report, just prevents escape (~5 dedicated registers).
- Reliable disassembly of variable-length x86 is a problem (a jump can land mid-instruction); fix by forcing all
  jump targets to 32-byte alignment. Cross-domain calls go through a read-only **jump table** of legal entry points.

---

## 9. Trusted execution environments (TEEs)

Confinement protects the system from an untrusted program; a TEE does the opposite — **protect sensitive code from
an untrusted system** (even a compromised OS/VMM), defending against software and probing attacks (not invasive
hardware attacks).

### 9.1 ARM TrustZone

- Each physical core provides two virtual cores — **Normal** and **Secure** worlds — time-sliced through a new
  **monitor mode**. The **NS bit** on the system bus reflects which world issued an access; the Secure world sees
  all resources, the Normal world only non-secure ones.
- World switch via the **SMC** instruction or certain hardware exceptions; monitor mode saves/restores state.
- **NS extends off-chip:** an NSTID bit on virtual addresses and NS bits in TLB/cache lines tag each entry by world,
  so secure and non-secure entries coexist; secure page tables can map normal memory (not vice-versa).
- Interrupts route through the monitor and can be steered to either world.
- Secure world typically runs a small secure OS (Qualcomm QSEE, Trustonic Kinibi, Samsung Knox, Genode).
- **Secure boot / chain of trust:** start from an inherently secure root (on-chip ROM / TPM / PUF); each stage
  verifies the next's signature (bootloader → secure OS → rich OS → trustlets).

### 9.2 Intel SGX (enclaves)

- **Reduced attack surface:** trust only the app + hardware; the OS, VMM and BIOS are untrusted.
- An **enclave** ("reverse sandbox") has its own code/data with controlled entry points; outside code — even the OS
  — cannot read enclave memory. Data is cleartext only inside the CPU package (registers/cache) and **auto-encrypted
  when it leaves** (the MEE).
- **Memory:** BIOS reserves **PRM**; the **EPC** (4 KB pages) holds enclave pages; **EPCM** (hardware-only) stores
  per-page access control; **SECS** holds per-enclave metadata.
- **Lifecycle instructions:** `ECREATE` (make SECS), `EADD` (add a page + record it in the measurement),
  `EEXTEND` (measure a 256-byte region; 16× per page), `EINIT` (verify the owner's SIGSTRUCT signature, allow
  entry), `EENTER`/`EEXIT` (user-mode enter/leave), **AEX** (asynchronous exit on interrupt: securely save state,
  push the AEP, leave enclave mode), `ERESUME`.
- **Attestation** via `MRENCLAVE` (SHA-256 of the build log): **intra-platform** (EREPORT + a report key shared by
  hardware and the target enclave, MAC-verified) and **inter-platform** (a Quoting Enclave signs with asymmetric
  keys for a remote verifier).
- SGX is itself broken by microarchitectural attacks (Foreshadow/L1TF, SGAxe, Plundervolt) — §11.

---

## 10. Hardware security

### 10.1 Physically unclonable functions (PUFs)

- Authenticate edge/IoT devices without stored keys: a PUF maps a **challenge → response** using nanoscale
  manufacturing variation, so no two chips agree. A **digital fingerprint** that can't be cloned or copied.
- Desired properties: **reliable** (same challenge, same chip → small difference despite temperature/noise/aging),
  **unique** (same challenge, different chips → large difference), **unpredictable**.
- **Silicon PUFs:**
  - **Ring-oscillator PUF**: frequency `f = 1/(2·n·t)` varies with process; compare two oscillators' counters →
    1 bit. (Virtex-4 results: ~59/128 bits differ inter-chip, ~0.6/128 intra-chip.)
  - **Arbiter PUF**: a race between two nominally symmetric paths latched by a D flip-flop; the challenge selects
    switch configurations. Vulnerable to ML modelling attacks.

### 10.2 Hardware Trojans

- Tiny, mostly passive circuits inserted at any stage (3rd-party IP, offshore fab, mask). Structure = **trigger**
  (rare event: a specific packet, elapsed time, a magic address like `0xdeadbeee`) + **payload** (flip a page's
  privilege, open a covert channel, fail the system).
- **Why stealthy:** a trigger on a wide input (e.g. a 32-bit comparator) has a tiny **control value** (≈ 2⁻¹⁶ per
  bit), so it almost never affects the output under random testing.
- **FANCI** (Waksman, CCS'13): compute each input's **control value** (truth-table influence on the output); inputs
  with near-zero control are suspicious backdoor triggers. Aim for no false negatives, tolerate false positives.
- **Detection:** optical (SEM/PICA — costly), functional testing (weak), **side-channel** (compare power/timing to
  a golden model). **Prevention / silencing** (Waksman, Oakland'11): power resets to flush ticking time-bombs;
  obfuscation; **sequence breaking** (reorder/insert events so a sequence trigger never sees its exact input);
  duplication (expensive).

### 10.3 Power analysis (side channels)

- A CMOS gate draws current only on a **0→1 / 1→0 transition**, so power ∝ switching activity, sampled at clock
  edges. Leakage models: **Hamming weight** (if a register is precharged) and **Hamming distance** (number of bit
  toggles between states).
- **SPA** reads secrets from a single trace (e.g. square-and-multiply in RSA). **DPA / CPA**: for each key guess,
  predict the leakage of an intermediate `C = F(P, K_guess)`, correlate the hypothesis against many measured traces,
  and the correct key gives the highest correlation (or the largest **difference-of-means** between the two buckets
  split on a bit of `C`). Statistical tools: correlation, mutual information, Bayes, difference-of-means.
- **Defences:** differential logic (hardware), **masking** (implementation), DPA-resistant ciphers (DRECON),
  re-keying.

---

## 11. Microarchitectural attacks

Performance optimisations break the abstractions we trusted for security (crypto, passwords, info-flow, rings,
ASLR, VMs, SGX/TrustZone).

- **Cache timing.** A victim's memory accesses change cache state, which the attacker times.
  - **Flush+Reload** (Yarom & Falkner): share a page (copy-on-write / shared libraries), `clflush` a line, let the
    victim run, then time a reload — fast = the victim used it. Leaks e.g. a square-and-multiply bit per iteration.
    Counters: no copy-on-write, restrict `clflush`, non-inclusive LLCs, fuzzy clocks, software diversification.
  - **Prime+Probe**: fill a cache set, let the victim evict some lines, time each set — slow sets reveal
    victim-touched sets (key-dependent table indices in AES). Works from JavaScript/WASM; cross-VM; keystroke
    timing; website fingerprinting.
  - **Internal/time-driven collision attacks**: cache hits when two cipher table indices collide reveal
    `K_i ⊕ K_j = P_i ⊕ P_j`.
- **Speculative execution.** Out-of-order + branch prediction execute **transient** instructions whose
  architectural effects are rolled back — but whose **microarchitectural** (cache) effects persist.
  - **Spectre v1 (bounds-check bypass):** train a branch, then with `x` out of bounds and `array_len` uncached, the
    CPU speculatively does `i = array[x]; y = array2[i*256]`, caching an `array2` line indexed by the secret → read
    via Flush+Reload. v2 (**branch-target injection**) poisons the BTB to redirect speculation to an attacker-chosen
    gadget.
  - **Meltdown:** a user read of a kernel address faults, but the transient window still does `y = array[i*256]`,
    leaking the kernel byte through the cache before the fault retires. Mitigation: **KPTI** (kernel page-table
    isolation).
  - Mitigations: Spectre v1 — `LFENCE` barriers, static analysis of injectable sites; v2 — per-process BTBs,
    retpolines, IBRS/IBPB; general hardware idea — speculative loads bypass the cache into a separate speculative
    buffer.
- **Rowhammer:** repeatedly activating a DRAM row flips bits in adjacent rows (a device-property attack, not a
  timing one) → privilege escalation. **DRAMA** uses the row buffer as a covert/side channel. Also **cold-boot**
  attacks and **fault injection**.

---
## 12. My assignments

Common setup: a Docker VM (`start.sh`) with **ASLR off** (`echo 0 > randomize_va_space`), **no PIE**
(`-no-pie`, fixed addresses), **no canary** (`-fno-stack-protector`), compiled `-m32 -ggdb -O0` for A1–A3.
Tools: GDB+GEF, objdump, readelf, Ghidra (A3), pwntools (A4). **Shell-byte constraint:** when invoking
`./bin "$(cat payload)"`, bash splits on `0x09` (tab), `0x0a` (newline), `0x20` (space); every address in a payload
must avoid those (and `0x00`, which `strcpy` truncates).

### 12.1 A1 — Buffer Overflow, bypass a custom canary (100: 50 exploit + 20 clean exit + 30 report)

- Two binaries: `nocanary` (demonstrated in class, payload given) and `withcanary` (self-implemented canary).
- `welcome(char *name)`: `char buf[12]`, `strcpy(buf, name)`, no bound check. The `withcanary` build stores a
  **hardcoded constant** `0xb0b28911` at `ebp-0xc`, and after `strcpy` does `cmpl $0xb0b28911, -0xc(%ebp); je ok;
  exit(1)`.
- **Stack layout** (from my disassembly): buf at `ebp-0x10`, canary at `ebp-0xc` (+4 from buf), saved ebx `ebp-4`,
  saved ebp `ebp`, **return at `ebp+4` (+20 from buf)**.
- **Bypass:** because the canary is a fixed constant visible in the disassembly, just place it at its offset so the
  check passes. Payload = 4 (fill buf) + 4 (canary `0xb0b28911`) + 12 (pad over saved ebx/ebp) + 4 (`exploit()`
  `0x08049196`) + 4 (`exit@plt` `0x08049070` for clean exit) = **28 bytes**.
- **Verified (Oct 2026):** the layout arithmetic and payload length check out; `payload_2` on disk is 28 bytes and
  matches `payloadCreate2.py`. The key insight — a hardcoded canary provides no real protection vs a random,
  NUL-containing, per-run compiler canary — is exactly the course's point.
- I correctly noted why you **cannot** skip the canary check by return-address manipulation: the check runs in the
  epilogue **before** `ret`, so controlling the return address alone is too late. Preserving the value is the right
  move.

### 12.2 A2 — Return-to-libc / ROP (100: 10 + 30 exploit + 2×10 clean exit + 40 report)

- 32-bit statically-linked `rop` binary, **NX on** (so no stack shellcode). `welcome()` has the same `strcpy`
  overflow; **24-byte** offset to the return slot (`buf` at `ebp-0x14`, ret at `ebp+4`). Hidden `winners(password)`:
  `strcmp(arg, "admin123456")`; on match prints the admin ID from global `id` (`.bss`); if `id == 100` (`0x64`)
  prints a hidden backdoor line.
- **Payload 1 (ret-to-libc):** overwrite ret → `winners()`, with `exit()` as its return address and the pointer to
  `"admin123456"` as its argument. Prints the ID, clean exit. (36 bytes.)
- **Payload 2 (ROP write-what-where):** must set `id = 100` first, but `0x00000064` has three NUL bytes, so it
  can't be embedded. **Trick:** the bytes `64 00 00 00` already exist at `0x080480f7` (ELF header region); a gadget
  chain *reads* the value from there and writes it to `&id`:
  1. `pop edx; pop ebx; pop esi; ret` → EDX = `&id` (0x080e3f1c);
  2. `pop esi; ret` → ESI = `0x080480f7`;
  3. `xchg eax, esi; ret` → EAX = `0x080480f7`;
  4. `mov eax, [eax]; ret` → EAX = `0x00000064`;
  5. `mov [edx], eax; ret` → `id = 100`;
  then fall through to `winners()` → `exit()`. All five gadget addresses are free of NUL/tab/newline/space. (72
  bytes.)
- **Verdict:** complete, correct, and the report explains each gadget's semantics (required for the 40 report
  marks). The "source a null-byte constant from inside the binary" idea is the clever part.

### 12.3 A3 — Format String (100: 20 + 30 exploit + 2×10 clean exit + 40 report)

- 32-bit dynamically-linked `vuln`, **NX off, partial RELRO (writable GOT)**. It lists `.txt` files in the CWD and
  does `printf(filename)` — the filename **is** the attacker-controlled format string. Reverse-engineered in
  **Ghidra**. Globals: `secret` (`.data`, 0x0804c060), hidden `flag()` (0x08049226), `putchar@GOT` (0x0804c030).
- **Payload 1 (read):** a `.txt` filename containing the address of `secret` plus a direct-parameter `%N$s`, so
  `printf` prints the secret string. (The offset N to reach the controlled slot was found by dumping `%x`s.)
- **Payload 2 (write → `flag()`):** overwrite `putchar@GOT` with `&flag` using `%hn` (two 16-bit half-writes to
  keep the printed-char counts small), so the program's next `putchar` jumps to `flag()`; returns to `main` → clean
  exit. (~45 bytes.) `putchar` was chosen because it's called reliably after the vulnerable `printf`.
- **Documented alternative:** combined format-string write + buffer overflow (overwrite a saved register / return
  target in `main`'s non-standard epilogue), ~224 bytes — less compact, needs extra writes for a clean exit. Good
  for the report to show understanding.
- **Verdict:** complete and correct, both tasks, with a second approach documented.

### 12.4 A4 — Heap Exploitation, glibc 2.23, no tcache (100: 20 + 40 + 40)

`vaultsrv`: a menu-driven vault manager. 8 vault structs in BSS at `0x6031e0` (each `0x20`: data ptr, size, state
flags, 16-byte name). Globals: `session_pin` (0x6032e8, random via `getrandom`), `admin_token` (0x6032f8, random),
`audit_trail` (0x6032f0). Evaluated by pwntools scripts that must trigger the bug and make the binary print the
flag (reading `flagN.txt` directly scores 0).

- **Stage 1 — Bin Anatomy (UAF fastbin leak, 20).** `vault_purge` calls `free()` but never clears the data pointer
  or the state flag → **use-after-free**; `vault_inspect` still `puts()` the freed chunk. Exploit: store two
  `0x18`-data (`0x20`-chunk) vaults, `purge(0)`, `purge(1)` (fastbin LIFO: V1→V0→NULL), `inspect(1)` leaks V1's
  **fd** = V0's chunk header; `vault0_data = fd + 0x10`; `claim(stage=1, value=vault0_data)`. Debug notes:
  `puts` stops at the two leading NUL bytes of a `0x0000555…` heap address (`ljust(8,'\x00')`), strips the trailing
  `\n`; local runs need `./ld-2.23.so` + `LD_LIBRARY_PATH=.`.
- **Stage 2 — Prev Size Lies (poison null byte → unlink, 40).** `vault_reseal` reads exactly `alloc_size` bytes
  then writes a NUL at `ptr + size` → **off-by-one**. With `alloc_size = 0x108` (chunk `0x110`) the NUL clears the
  next chunk's PREV_INUSE and turns `0x111` into `0x100`. Exploit: craft a **fake chunk** inside V0's data whose
  `fd = &vaults[0].data − 0x18`, `bk = &vaults[0].data − 0x10` so the safe-unlink check passes
  (`fd->bk == bk->fd == P`); place a fake nextchunk header (prev_size `0x100`, size `0x21`) inside V1 to pass the
  `size == next->prev_size` check; `reseal(0)` poisons V1's size; `purge(1)` coalesces backward and unlinks → V0
  and V1 overlap; a new `store(4, 0x150)` writes `0xC0FFEE1337` into V1's old region; `claim(stage=2, …)`.
- **Stage 3 — Unlink Undone (BSS, 40).** `session_pin` is 8 random bytes (`getrandom`), and the stage-3 `claim`
  check is `value == session_pin`. **My solution leaks it** via `vault_audit`, which does a double dereference
  `val = *(*(vaults[i].data))` and prints the full 8-byte value as hex — a read primitive reached by pointing a
  vault's data at `&session_pin` through the Stage-2 overlap. Submit the leaked value.
  - **Observation / deviation:** the spec frames Stage 3 as *overwrite* a random BSS value (via the unsafe-unlink
    arbitrary write, preserving a critical pointer with an unsorted-bin libc leak). My route is a **leak**, not a
    write. Because the grader only checks `value == session_pin`, the leak passes; but if a future grader checks
    that the value was *written by the exploit*, this stage would need the unlink-write version. Worth re-reading
    before relying on it.
- **Verdict:** all three stages solved; strong heap-layout diagrams and honest debugging notes. The Stage-3
  interpretation is the one thing to double-check.

---

## 13. My project — AI-agent security / indirect prompt injection

**Title:** "Analyzing AI Task Deployment Workflows and Authentications Involved." Milestones: literature survey →
demo → final report + slides. This is the most directly Phase-3-relevant artifact in all my coursework.

### 13.1 The argument

- LLMs moved from **oracle** (writes text) to **actor** (calls tools: send email, query DB, move money). The
  "human speed bump" between output and consequence is removed by design, so an old bug (prompt injection) is now
  consequential (a tool call, not regrettable text).
- **Agentic stack:** raw model (two files: weights + inference code) → augmentation (prompt engineering, fine-tune,
  **RAG**, **tools**) → deployment levels: L1 vanilla, L2 fixed workflow, L3 agent (model decides tool calls),
  **L4 ReAct** (reason → act → observe loop; each of the three is an attack surface; the observe step is where
  injected content arrives).
- **Key architectural fact:** attention has **no notion of trust** — a system-prompt token, a user token and a
  token from a malicious resume the agent just read are the same kind of object. This is why prompt injection is
  structural, not a bug to patch.

### 13.2 Threat model (three layers) and the OWASP Top-10-for-LLM 2025 mapping

- **Input/prompt layer:** LLM01 Prompt Injection (direct and **indirect**), LLM07 System-Prompt Leakage.
- **Action layer:** LLM06 Excessive Agency (blast radius), LLM05 Improper Output Handling, plus the **confused
  deputy** — the agent legitimately holds the user's OAuth tokens and is tricked into misusing them.
- **Data / supply-chain layer:** LLM02 Sensitive-Info Disclosure, LLM03 Supply Chain, LLM04 Data/Model Poisoning,
  LLM08 Vector/Embedding Weaknesses (RAG store), LLM09 Misinformation (hallucinations the agent acts on).

### 13.3 The demonstration

- **Vulnerable agent** (`agent_vulnerable.py`): a ReAct agent on **Ollama (llama3.1)** + LangChain with `read_file`
  and `send_email` tools. User asks it to "summarise `resume.txt`"; the resume contains a hidden instruction to read
  `company_secrets.txt` and email it to an attacker. The agent obeys → `[EXFILTRATION SUCCESSFUL]`. Textbook
  **indirect prompt injection**: malicious instructions enter through *data* the agent reads, not the user prompt.
- **Mitigation 1 — Human-in-the-loop** (`agent_mitigated_hitl.py`): require explicit human approval before any
  privileged action (external email, data modification, money). Scales linearly in privileged actions; strong but
  high friction.
- **Mitigation 2 — Dual-LLM Inspector** (`agent_mitigated_inspector.py`, after Willison): a second, stricter LLM
  with **no tool access** vets each tool call ("is this exfiltration? ALLOW/REJECT") before execution.
- A Streamlit dashboard (`dashboard*.py`) runs vulnerable vs mitigated side by side.

### 13.4 The honest limits and the proposal

- **Formal impossibility:** deciding whether arbitrary text is an injection is undecidable in general; no inspector
  is a silver bullet. The dual-LLM inspector is itself an LLM and can be meta-injected.
- **Proposed LLM Security Gateway:** four checkpoints — **S1 input, S2 retrieved content, S3 output, S4
  inter-agent** — each a small model with a narrow, codified policy and no tools of its own, intercepting the attack
  path at each boundary an agent crosses. Research agenda: adaptive adversaries, memory poisoning, long-horizon goal
  drift, and the meta-injection problem.
- Related external work worth citing in Phase 3: Greshake et al. (indirect prompt injection, 2023), Willison's
  dual-LLM pattern, Google **CaMeL** ("Defeating Prompt Injections by Design", 2025, capability-based), Microsoft
  **FIDES** (information-flow control for agents, 2025), **spotlighting** (Microsoft).

---

## 14. Revision Q&A

- *Why doesn't W^X stop ROP/ret-to-libc?* They execute only existing code pages, never writable data.
- *Why doesn't a canary stop a buffer overread (Heartbleed)?* No write occurs, so the canary is unchanged. ASLR and
  W^X also don't help (no segment escape, no code execution).
- *How does `%n` give arbitrary write?* It stores the running printed-char count to a pointed-to address; field
  widths control the count, `%hn` splits a 32-bit write into two 16-bit writes, and direct params `%N$` pick the
  stack slot holding the target address.
- *Why is a GOT entry a good write target?* Partial RELRO leaves it writable, and it is dereferenced on the next
  PLT call → control hijack without touching the stack.
- *How does the fastbin UAF leak a heap address?* A freed fastbin chunk's first 8 bytes become the `fd`; printing
  the freed chunk reveals a heap pointer.
- *Why does a single NUL byte matter on the heap?* Clearing PREV_INUSE and shrinking the size field makes `free`
  coalesce backward into a crafted fake chunk → chunk overlap.
- *How is the safe-unlink check bypassed?* Point `fd`/`bk` at a location that already stores the chunk's own
  address, so `fd->bk == P` and `bk->fd == P` hold.
- *No-read-up vs no-write-down?* BLP (confidentiality): no read up, no write down. Biba (integrity): no read down,
  no write up.
- *ACL vs capability?* ACL = list on the object (easy revoke, harder delegate); capability = ticket held by the
  subject (easy delegate, harder revoke). A Unix fd is a capability.
- *Segment matching vs address sandboxing (SFI)?* Matching *checks* and can report the fault; sandboxing *enforces*
  the segment-ID bits (faster, no report).
- *Spectre vs Meltdown?* Spectre mistrains prediction to make the victim leak its own memory via transient,
  cache-visible accesses; Meltdown uses a transient window after a faulting kernel read. KPTI fixes Meltdown;
  barriers/retpolines/per-process BTBs fix Spectre.
- *Why are x86 ROP gadgets plentiful but ARM/RISC ones scarce?* x86 is dense and variable-length, so unaligned
  decoding yields unintended gadgets; fixed-width aligned ISAs don't.
- *What makes a hardware Trojan trigger stealthy (FANCI)?* Its inputs have near-zero **control value** over the
  output, so random testing almost never activates it.
- *HD vs HW power model?* Hamming distance counts bit toggles between two states; Hamming weight counts set bits
  (valid when the bus/register precharges).

---

## 15. Bridges to AI infrastructure

1. **The project is the seed of a Phase-3 flagship.** Harden the LLM Security Gateway into a real
   information-flow-control layer for agents (taint tool outputs as untrusted, propagate labels through the ReAct
   trace, block privileged sinks on tainted args). This fuses CS6570 (confinement, info-flow, TEEs) with CS6843
   (taint/slicing) and is a hot, publishable area (CaMeL, FIDES).
2. **Memory-safety analysis for ML/GPU code.** SoftBound-style bounds instrumentation and the heap/stack discipline
   here transfer to catching OOB/races in generated CUDA/Triton kernels (ties to CS6843 A4 and CS6023).
3. **Side-channel / microarchitectural awareness for inference.** Cache-timing and Flush+Reload matter for
   multi-tenant inference servers and for protecting model weights/prompts; constant-time kernels and partitioned
   caches are infrastructure concerns.
4. **TEE-backed confidential inference.** SGX/TrustZone + attestation underpin "confidential AI" (run a model or
   handle sensitive prompts where the host is untrusted) — a concrete systems+AI resume project.
5. **Confinement for agent sandboxes.** OKWS/SFI/seccomp/gVisor patterns are exactly how you safely run
   LLM-generated or agent-invoked code (the `read_file`/`send_email` tools in my demo are the attack surface).
6. **Supply-chain and model/vector-store integrity.** LLM03/LLM04/LLM08 map onto securing training data,
   model artifacts and RAG indices — MLOps security.

---

## 16. References

- **Course / practice:** CS6570 slides (Chester Rebeiro); the NPTEL "Secure Systems Engineering" modules in
  `Practice/casl-sse-*` (ELF, buffer overflows, ROP, ASLR, format strings, heap, access control, SFI, TEEs,
  hardware, microarchitectural attacks).
- **Binary exploitation:** Aleph One, "Smashing the Stack for Fun and Profit", Phrack 49 (1996);
  Shacham, "The Geometry of Innocent Flesh on the Bone" (ROP), CCS'07; scut/team-teso, "Exploiting Format String
  Vulnerabilities" (2001); Solar Designer, unsafe-unlink; `shellphish/how2heap`; sploitfun / Azeria glibc-heap
  series; Phrack 60:10 (integer overflows).
- **Defences:** Cowan et al., StackGuard (canaries), USENIX Security'98; PaX/ASLR; Eli Bendersky, "Position
  Independent Code (PIC)"; Nagarakatte, Zhao, Martin, Zdancewic, **SoftBound**, PLDI'09.
- **Famous bugs:** Heartbleed (CVE-2014-0160); Stagefright (Drake, 2015); Code Red (2001).
- **Access control / MAC:** Lampson, "Protection" (1971); Harrison–Ruzzo–Ullman (1976); Bell & LaPadula (1974);
  Biba (1975); Sandhu, "Lattice-Based Access Control Models" (1993).
- **Confinement / SFI:** Wahbe, Lucco, Anderson, Graham, **SFI**, SOSP'93; Krohn, **OKWS**, USENIX'04; Google NaCl.
- **TEEs:** ARM TrustZone whitepaper; McKeen et al., **SGX**, HASP'13; Anati et al., SGX attestation, HASP'15.
- **Hardware:** Devadas et al., PUFs, DAC'07; Waksman & Sethumadhavan, **FANCI** (CCS'13) and "Silencing Hardware
  Backdoors" (Oakland'11); Kocher, Jaffe, Jun, **DPA** (CRYPTO'99).
- **Microarchitectural:** Yarom & Falkner, **Flush+Reload** (USENIX Sec'14); Osvik–Shamir–Tromer, Prime+Probe;
  Kocher et al., **Spectre**; Lipp et al., **Meltdown** (both USENIX Sec'18); Kim et al., **Rowhammer** (ISCA'14).
- **AI-agent security (project):** 2025 OWASP Top 10 for LLM Applications; Greshake et al., indirect prompt
  injection (2023); Willison, dual-LLM pattern; Debenedetti et al., **CaMeL** (2025); Microsoft **FIDES** (2025);
  Yao et al., **ReAct** (2023).
