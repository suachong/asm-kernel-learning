# Terminology Primer: Reading a GPU `.s` File

> **Purpose:** Every term in the vendor's READMEs (`s_setprio`, `buffer_load
> hoisting`, `vmcnt`, MFMA, occupancy, `ds_write drain`, etc.) explained from
> first principles, with concrete pointers into the `.s` files you have.
>
> **You should be able to:** after reading this, look at any 10-line slice of
> `~/ggemm-asm-fwd/kernels/combined_v5.s` and explain in plain English what
> each instruction is doing and why.

---

## 1. The mental model: how a GPU executes a kernel

### MI355X hardware spec table (the one number sheet)

Source: AMD ROCm GPU hardware specifications,
`https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html`.

| Resource | MI355X (CDNA4, gfx950) | Notes |
|---|---|---|
| **HBM (VRAM)** | **288 GiB HBM3e** | Per GPU |
| **Compute units (CUs)** | **256** | Organized as 8 XCDs × 32 CUs |
| **SIMDs per CU** | 4 | A wavefront executes on one SIMD |
| **Wavefront width** | 64 threads | |
| **LDS per CU** | 160 KiB | "Local Data Share"; NVIDIA's "shared memory" |
| **L1 vector cache per CU** | 32 KiB | Caches HBM loads |
| **L1 scalar cache** | 16 KiB per 2 CUs | Caches scalar (s\_) loads |
| **L1 instruction cache** | 64 KiB per 2 CUs | Holds the kernel `.text` |
| **L2 cache** | 32 MiB total | **4 MiB per XCD**, shared by 32 CUs in that XCD |
| **L3 (Infinity Cache)** | 256 MiB total | Shared across all 8 XCDs |
| **VGPR file per CU** | 512 KiB storage | Per-wavefront addressable: **512 VGPRs** |
| **SGPR file per CU** | 12.5 KiB storage | Per-wavefront addressable: **~102 SGPRs** |
| LLVM target | `gfx950` | |
| GFXIP version | 9.5 | |

For comparison (so the lineage is clear):

| GPU | Arch | LLVM | HBM | CUs (per XCD) | LDS/CU |
|---|---|---|---|---|---|
| MI355X | CDNA4 | gfx950 | 288 GiB | 256 (32 × 8) | 160 KiB |
| MI325X | CDNA3 | gfx942 | 256 GiB | 304 (38 × 8) | 64 KiB |
| MI300X | CDNA3 | gfx942 | 192 GiB | 304 (38 × 8) | 64 KiB |

Notice that MI355X has *fewer* CUs than MI300X (256 vs 304) but more LDS per CU (160 vs 64 KiB) and the new MFMA opcodes — i.e. each CU is more capable. That's why per-shape kernel tuning matters: the cross-generation tradeoffs are not uniform.

### What the chiplets actually are (XCD, compute die, MCM)

A "GPU" on MI355X is a **multi-chip module (MCM)** — a single package
containing many pieces of silicon glued together with AMD's Infinity
Fabric:

- **8 XCDs (eXtension Compute Dies)**. Each XCD is one piece of silicon
  containing **32 CUs + 4 MiB of L2 cache** dedicated to those 32 CUs.
  CUs *within* an XCD share that L2. CUs *across* XCDs do not — cross-XCD
  traffic goes through the 256 MiB L3 (Infinity Cache) and the Infinity
  Fabric switches, which is significantly slower than the local L2.
- **I/O dies** holding the HBM3e memory controllers, PCIe, and Infinity
  Fabric switches.
- **8 HBM3e stacks** providing the 288 GiB of VRAM, bonded to the I/O
  dies.

"Compute die" just means "a chiplet containing CUs." On MI355X the
compute die *is* the XCD. (Older monolithic GPUs had a single compute
die = the whole chip. AMD switched to chiplets with MI300 because it lets
them stitch together transistor counts that wouldn't fit on a single
piece of silicon.)

**Why XCD locality matters for your tiling.** When two consecutive tiles
of a GEMM both run on CUs *within the same XCD*, they share that XCD's
4 MiB L2 — the second tile gets cache hits on the operand data the first
one loaded. Across XCDs there's no sharing; every CU pays the full round
trip to L3 or HBM. Your intuition is right: when designing a tile
schedule, keep consecutive tiles within an XCD whenever possible. That's
exactly what `grouped_vark_dot_scaled.py:36-41` does — it reshuffles
`program_id`s so workgroup `pid` and `pid+1` land on CUs in the same XCD.

### CU internals (where the compute actually happens)

Each CU contains:

- **4 SIMDs** (Single Instruction Multiple Data units) — the actual
  vector execution units inside the CU. **Each SIMD is 16 physical lanes
  wide.** A wavefront has 64 threads but runs on one SIMD, so the SIMD
  time-multiplexes the wave over **4 cycles**: 16 threads per cycle until
  all 64 threads have executed the instruction. This is AMD's
  "quarter-wave issue" model (inherited from GCN). With 4 SIMDs per CU,
  the CU has 4 × 16 = 64 physical execution lanes total and can advance
  up to 4 wavefronts simultaneously (one per SIMD). Many more wavefronts
  can be *resident* — the SIMD context-switches between them every 4
  cycles to hide latency.

  *Exception: MFMA.* `v_mfma_*` instructions use the matrix-core pipeline
  inside the SIMD, which consumes the full wavefront at once and takes
  ~32 cycles to complete. One MFMA does the work of dozens of regular
  VALU ops, which is why MFMA dominates the kernel's runtime profile.
- **VGPR file (512 KiB)** — register storage shared across all resident
  wavefronts. A wavefront can address up to 512 32-bit VGPRs, and one
  VGPR's storage cost per wavefront is 64 lanes × 4 B = 256 B. **More
  VGPRs per wave → fewer resident waves → lower occupancy.** This is the
  knob that matters most for ASM tuning.
- **SGPR file (12.5 KiB)** — scalar register storage. A wavefront
  addresses ~102 SGPRs.
- **LDS (160 KiB)** — shared SRAM. A workgroup's threads collaborate
  through LDS.
- **L1 vector cache (32 KiB)** — caches HBM loads.

### SIMDs are NOT hardware queues

This question comes up because both have "lanes/queues" vibes. They're at
completely different layers of the stack:

| Concept | Lives where | What it does | How many on MI355X |
|---|---|---|---|
| **SIMD** | Inside a CU | Hardware execution unit. A wavefront runs on one SIMD; the SIMD executes that wave's instruction across 64 lanes in parallel each cycle. Pure compute hardware. | 4 per CU × 256 CUs = 1024 |
| **Hardware queue** (HSA / ACE queue) | Top-level GPU command processor | A FIFO of kernel launches that the host (CPU/HIP) submits to the GPU. The Command Processor pulls launches from these queues and dispatches workgroups to CUs. | Small fixed count (a handful per ACE; ACEs serve all CUs) |

Analogy: SIMDs are **ALUs inside the engine**; hardware queues are
**work orders the engine pulls from**. A HIP stream backs a hardware
queue. When you call `hipModuleLaunchKernel`, the launch lands in a
queue, the Command Processor dispatches workgroups across CUs, each
workgroup is decomposed into wavefronts, and each wavefront is assigned
to one SIMD on one CU.

Practically: you never tune at the SIMD level directly. You tune at the
wavefront level (VGPR/SGPR usage, LDS, occupancy), and the hardware
places waves onto SIMDs.

### Concrete example: occupancy math for our FWD kernel

The vendor's `combined_v5.s` reports (`~/ggemm-asm-fwd/README.md:118`):

> 248 VGPRs, 55 SGPRs, 65536 LDS, 512 threads/WG

Reading this against the table above:

- **VGPRs**: 248 of 512 per wavefront. Storage cost per resident wave =
  248 × 256 B = 63,488 B ≈ 62 KiB. Per SIMD storage = 128 KiB, so each
  SIMD can hold at most ⌊128 KiB / 62 KiB⌋ = 2 resident waves. But the
  CU also has to fit them all in 512 KiB, and 512 threads/WG = 8 waves
  per workgroup. 8 waves × 62 KiB = 496 KiB — just barely under 512 KiB.
  Exactly **1 workgroup fits per CU** = "occupancy 1."
- **SGPRs**: 55 of ~102, no constraint.
- **LDS**: 65,536 B = 64 KiB of the available 160 KiB per CU. Could fit
  2 workgroups by LDS alone, but VGPRs limit us to 1.
- **Threads/WG**: 512 = 8 waves × 64 threads.

That's where the README's "Occupancy 1 → 86% stall-bound" comes from.
The kernel is at the edge of the VGPR budget; one fewer wave per CU
means less ability to hide memory latency by swapping to another wave.

### Worked example: occupancy 1 vs occupancy 2 cycle timeline

Each SIMD has ~4 independent execution pipelines that can dispatch in
parallel if you give them independent instructions:

```
SIMD
├── Std VALU       → regular v_* instructions, ~4 cyc each
├── Matrix Core    → v_mfma_*, ~32 cyc each, dedicated tensor engine
├── LDS unit       → ds_read / ds_write, ~20 cyc each
└── VMEM unit      → buffer_load / buffer_store, ~300+ cyc each
```

The whole game of ASM scheduling is **keeping all four pipelines busy
simultaneously** by interleaving independent instructions across them.

Stylized one-K-iteration timeline. Setup: a `buffer_load`, then 4
dependent MFMAs, then a `ds_write`.

**Occupancy 1 (only W0 resident on SIMD 0):**

```
cycle  | Std VALU             | Matrix Core   | LDS unit      | VMEM unit
-------|----------------------|---------------|---------------|----------------
   0   |                      |               |               | W0: buffer_load v[0:3]
   4   | W0: s_waitcnt vmcnt(0) → STALL. No other wave to swap. SIMD idle.
 5-303 | -- idle ~300 cycles --                                ← 300 wasted cycles
 304   |                      |               |               | W0 load completes
 305   |                      | W0: MFMA #1   |               |
 309   |                      | W0: MFMA #2   |               |
 313   |                      | W0: MFMA #3   |               |
 317   |                      | W0: MFMA #4   |               |
 321   |                      |               | W0: ds_write  |
 325   | W0: loop ctrl, branch back
```

~92% of cycles are idle waiting on the HBM load. This is the
"86% stall-bound" the README cites.

**Occupancy 2 (W0 and W4 both resident on SIMD 0):**

```
cycle  | Std VALU             | Matrix Core   | LDS unit      | VMEM unit
-------|----------------------|---------------|---------------|----------------
   0   |                      |               |               | W0: buffer_load
   4   |                      |               |               | W4: buffer_load
   8   | W0: s_waitcnt vmcnt(0) → W0 STALLED. SIMD picks W4.
  16   |                      | W4: MFMA      |               |   ← carry-over MFMAs
  20   |                      | W4: MFMA      |               |   from W4's previous iter
  24   |                      | W4: MFMA      |               |
  28   |                      | W4: MFMA      |               |
  32   |                      |               | W4: ds_write  |
  ...  | SIMD ping-pongs between W0 and W4 as loads complete
 304   |                      |               |               | W0 load completes
 308   |                      | W0: MFMA #1   |               |
 312   |                      | W0: MFMA #2   |               |
 ...
```

The 300 idle cycles of occupancy 1 became 300 cycles of W4's useful
work. That's "latency hiding by waveswapping." The matrix core keeps
running at full throughput because whenever one wave is stalled on HBM,
the other has independent MFMAs ready.

### The optimization techniques mapped to the timeline

Now the README's optimizations make sense as targeted attacks on the
occupancy-1 picture:

| Technique | Timeline effect |
|---|---|
| **Buffer-load hoisting** | Move the `buffer_load` from cycle 0 to a much earlier cycle (e.g. ~3000 cycles before its use, in the previous K-iter). By the time the wave hits `s_waitcnt vmcnt(0)`, the load has long since completed. Recovers most of the 300 idle cycles even at occupancy 1. |
| **MFMA + ds_write interleaving** | Issue MFMA into the Matrix Core column at the same cycle as ds_write into the LDS unit column. Separate hardware; don't serialize them. |
| **`s_setprio 3/0`** | Bias the scheduler's tiebreaker when ≥ 2 waves are simultaneously ready. Mostly relevant at occupancy ≥ 2. |
| **MOV / waitcnt removal** | Each removed instruction = 4 fewer cycles in the Std VALU column. Doesn't fix stalls, but makes each iteration shorter. |

The mental model to carry whenever you read an agent's diff: **which
pipeline column did the diff touch, and did it shorten or lengthen the
cycle count there?** If the agent can't answer that question, the diff
is a guess.

A **kernel** is a program that runs on every CU in parallel. Each CU runs
some number of **wavefronts** simultaneously — that count is the kernel's
**occupancy**. Occupancy 1 means one wavefront per CU at a time. Higher
occupancy = more parallelism = better latency hiding.

### The optimization economy

99% of GPU optimization is one sentence: **the MFMA (matrix multiply) units
are starving for data, and you're trying to feed them faster than they can
consume.**

To do that you must:

1. Issue HBM loads (slow) early enough that the data is ready when MFMA
   needs it.
2. Stage data through LDS (fast, but limited) without making LDS the new
   bottleneck.
3. Keep the wavefront issuing instructions, not stalling on `s_waitcnt`.
4. Use the densest MFMA opcode the hardware supports (more FLOPs/instruction).

The vendor's optimizations in `~/ggemm-asm-fwd/README.md` and
`~/ggemm-asm-wgrad/README.md` are all instances of (1), (2), and (3). The
Triton compiler picks the opcode for (4).

---

## 2. The instruction families you'll see

Look at the first 50 lines of `~/ggemm-asm-fwd/kernels/combined_v5.s`. You
see prefixes like `s_`, `v_`, `buffer_`, `ds_`. Here's the cheat sheet:

| Prefix | Family | Meaning |
|---|---|---|
| `s_` | Scalar | One operation across the whole wavefront. Uses SGPRs. Cheap. Examples: `s_load_dwordx2`, `s_branch`, `s_waitcnt`, `s_barrier`. |
| `v_` | Vector | Per-thread operation. Uses VGPRs. The bulk of compute. Examples: `v_mov_b32`, `v_add_u32`, `v_mfma_*`. |
| `buffer_` | Global memory | Loads/stores to HBM (slow, ~300+ cycles). Examples: `buffer_load_dwordx4`, `buffer_store_short`. |
| `ds_` | LDS | Loads/stores to shared memory (fast, ~20 cycles). Examples: `ds_read_b128`, `ds_write_b128`. |
| `flat_` | Generic memory | Address-decoded: goes to HBM or LDS depending on the pointer. Slower than `buffer_` when you know it's HBM. |

### MFMA — the workhorse instruction

`v_mfma_f32_16x16x128_f8f6f4 acc[0:3], v[0:3], v[4:7], acc[0:3]` does a
**16×16 outer product over 128 K-elements** in fp8, accumulating into 4
fp32 VGPRs (a 16×16 tile of fp32 outputs per thread group). One MFMA
issues to a separate matrix-core pipeline and takes roughly 32 cycles to
complete (verify exact throughput in the CDNA4 ISA MFMA table).

Field-by-field decoding of the mnemonic:

| Token | Meaning |
|---|---|
| `v_mfma_f32` | Vector Matrix-Fused-Multiply-Add, fp32 outputs |
| `16x16x128` | **M=16, N=16, K=128** for this instruction |
| `f8f6f4` | Inputs may be FP8, FP6, or FP4 (gfx950 "flex-dtype" family) |
| `acc[0:3]` (1st) | Destination: 4 VGPRs holding per-thread fragment of 16×16 fp32 output |
| `v[0:3]` | A operand fragment |
| `v[4:7]` | B operand fragment |
| `acc[0:3]` (2nd) | Input accumulator (same regs as dest → FMA semantics: `acc = A @ B + acc`) |

**FLOP count formula: 2 × M × N × K.** (1 multiply + 1 add per MAC, summed
over all output elements × the K reduction.) So:

- `v_mfma_f32_16x16x128_f8f6f4`: 2 × 16 × 16 × 128 = **65,536 FLOPs/instr**
  (FWD/DGRAD in `combined_v5.s`)
- `v_mfma_f32_32x32x64_f8f6f4`: 2 × 32 × 32 × 64 = **131,072 FLOPs/instr**
  (dot_scaled wgrad in `dot_scaled_v2.s`) — 2× the per-instruction work
  of the FWD MFMA
- `v_mfma_f32_16x16x32_fp8_bf8`: 2 × 16 × 16 × 32 = **16,384 FLOPs/instr**
  (legacy wgrad in `variable_k_wgrad_mega.s`) — the older CDNA3-era MFMA;
  4× less per-instruction work than the new f8f6f4 family.

If you ever see a `v_mfma_*` and want to know its throughput, look it up in
the CDNA4 ISA PDF's MFMA opcode table.

### `buffer_load_*` — the latency you're hiding

`buffer_load_dwordx4 v[0:3], v100, s[8:11], 0` loads 16 bytes (4 dwords)
from HBM at the address computed from `v100` + `s[8:11]` into VGPRs
`v[0:3]`. **The instruction doesn't wait for the data.** It issues, and the
data arrives in `v[0:3]` ~300+ cycles later. You wait for it with
`s_waitcnt vmcnt(N)` (see below).

This decoupling is the entire reason "buffer_load hoisting" works:

```asm
buffer_load_dwordx4 v[0:3], ...        ; issue load. Data NOT ready yet.
v_mfma_f32_*  ...  ...  ...            ; do 50 MFMAs. Compiler can choose
v_mfma_f32_*  ...  ...  ...            ;   to overlap them with the load.
v_mfma_f32_*  ...  ...  ...
... (~300 cycles of MFMA later) ...
s_waitcnt vmcnt(0)                     ; now stall until the load finishes
v_mfma_f32_*  v[0:3], ...              ; use the loaded data
```

If you instead issue the load right before the `s_waitcnt`, you spend ~300
cycles stalled doing nothing. That's the "+10%" win in the wgrad kernel:
moving the load earlier in the program text changes when it executes
relative to MFMAs.

### `ds_read` / `ds_write` — the LDS staging

LDS is the bridge between HBM and the MFMA registers. A common pattern:

```
buffer_load HBM → VGPR  (slow, async)
ds_write    VGPR → LDS  (fast, async; needs lgkmcnt to wait)
ds_read     LDS → VGPR  (fast, async)
v_mfma      VGPR → VGPR  (compute)
```

Double-buffering ping-pongs two LDS regions so the MFMAs always have data
in flight. The "ds_write drain" in `combined_v5.s` is the sequence of 8
`ds_write`s that flush the current buffer to LDS between K-iterations.

### `s_waitcnt vmcnt(N)` / `lgkmcnt(N)` — the synchronization primitives

The hardware tracks in-flight memory operations with counters:

- **`vmcnt`** — number of outstanding VMEM (HBM) loads/stores not yet
  complete. `s_waitcnt vmcnt(0)` means "stall until all outstanding HBM
  ops are done." `s_waitcnt vmcnt(3)` means "stall until at most 3 are
  outstanding" — used to drain in **FIFO order** when you've issued more
  loads than you want hanging.
- **`lgkmcnt`** — same idea but for LDS, GDS, KM (scalar mem). `s_waitcnt
  lgkmcnt(0)` waits for all LDS reads/writes to complete.
- **`expcnt`** — export (output to graphics; not relevant here).

These are the most-frequent instructions in the `.s` file. **Removing a
redundant `s_waitcnt` is one of the cheapest possible optimizations** (one
fewer stall point) and is exactly what bullet 2 of the dot_scaled wgrad
optimizations does.

### `s_barrier` — the workgroup sync

Forces all wavefronts in a workgroup to reach this point before any
continues. Expensive. The combined_v5 kernel has 8 barriers per K-loop
iteration; the README cites them as a 14% structural overhead.

### `s_setprio 0..3` — the wave-priority hint

`s_setprio 3` tells the CU's instruction scheduler "this wave has higher
priority than other waves on this CU; prefer to schedule its instructions."
`s_setprio 0` is "normal/low priority."

In `combined_v5.s` the pattern is `s_setprio 3` right after `s_barrier`
(start of the MFMA-heavy section) and `s_setprio 0` right before
`buffer_load` (yield while waiting for HBM). Net effect: the MFMA wave
gets favored when there's work to do, and other waves get to run during
memory stalls. **rocBLAS uses the same trick** — it's not magic, it's a
scheduler hint.

This is only relevant when occupancy > 1 (there's competition for the CU).
At occupancy 1 it's nearly free but doesn't help either, which is why the
README shows "+0.1-0.3%" and notes "no benefit at occupancy=2" in the
opposite direction depending on context.

---

## 3. The optimization techniques, decoded

### "Buffer_load hoisting"

Take a `buffer_load_dwordx4` instruction and move it **earlier in the
program text**, so the load issues sooner and has more time to complete
before its result is needed. Constraints:

1. The destination VGPRs must be **dead** (not holding live data) at the
   new location. The README explicitly identifies the dead-VGPR set:
   `v[238:245]` in fwd, `v[216:223]` in wgrad legacy.
2. The address registers used by the load (e.g. `v194`, `v193` for fwd)
   must hold valid values at the new location.
3. The vmcnt-wait at the consumer must still drain the load FIFO in the
   right order (loads complete in issue order; `vmcnt(3)` drains 3 oldest
   loads first).

When the README says "extending HBM latency cover from ~120 to ~3500
cycles," that's "before hoisting the load issues 120 cycles before its
use; after hoisting, 3500 cycles before its use." 3500 cycles is comfortably
more than the ~300-cycle HBM round-trip, so the load fully overlaps with
compute and the `s_waitcnt vmcnt(0)` becomes a no-op.

### "MFMA / ds_write interleaving with drain"

Default Triton output groups all `ds_write`s back-to-back, then runs all
remaining MFMAs:

```
ds_write B0   ds_write B1   ...   ds_write B7
MFMA #1   MFMA #2   ...   MFMA #4
```

But each `ds_write` requires draining a vmcnt-wait (the buffer was loaded
from HBM, written to LDS). The drain waits are stall cycles. If the
tail MFMAs are **independent of the `ds_write` source registers** (i.e.
they read different VGPRs), you can move them into the stall windows:

```
MFMA #1
ds_write B0   ds_write B1     ← these drain during MFMA #1's 32-cycle execution
MFMA #2
ds_write B2   ds_write B3
... etc
```

The 4 tail MFMAs go from costing 4 × ~30-cycle stall to ~0 (they overlap
the drain). The +0.6-0.8% is that recovered stall.

### "Redundant MOV / waitcnt elimination"

Triton emits things like `v_add_u32_e32 v200, 0, v201` (add 0 to v201,
store in v200) — that's just a copy. If you can show the destination
register isn't read between the copy and the next overwrite, the copy is
dead and can be removed. Same for back-to-back `s_waitcnt lgkmcnt(0)`
with no new LDS ops between them — the second one waits on nothing.

Each removal saves one instruction issue slot, ~1 cycle. Many small wins
compound (the wgrad kernel got +0.3% from removing 6 MOVs).

### "ds_read overlap"

ds_read has ~20-cycle latency. If two ds_reads share a destination VGPR
("ds_read v[10:11]; use v[10:11]; ds_read v[10:11]; use v[10:11]"), you
can't issue the second until the first's use is done. But if you have a
**dead pair of VGPRs available** (the wgrad README mentions `v[138:139]`
which were holding scale factors no longer needed), you can ping-pong:

```
ds_read v[10:11]
ds_read v[138:139]    ← issue immediately, alternative destination
use v[10:11]          ← first ds_read result ready ~20 cycles later
use v[138:139]        ← second ds_read result ready ~20 cycles after that
```

Both ds_reads are in flight simultaneously, hiding their latency behind
each other.

### "Loop tail restructure"

The very last MFMA of the K-loop can co-execute with the loop control
instructions (`s_add_i32` to bump the counter, `s_cbranch` to loop back).
If you reorder so the loop control sits in the MFMA's 32-cycle issue
window, you save the loop-control cycles entirely.

---

## 4. How to read the `.s` file

Look at `~/ggemm-asm-fwd/kernels/combined_v5.s` line 8 onwards:

```asm
s_load_dwordx2 s[2:3], s[0:1], 0x0       ; load 2 scalars (lhs ptr) from kernarg
s_load_dwordx8 s[4:11], s[0:1], 0x8      ; load 8 scalars (rhs ptr, out ptr, etc)
s_load_dwordx4 s[12:15], s[0:1], 0x28    ; load 4 more scalars
s_waitcnt lgkmcnt(0)                     ; wait for all scalar loads to finish
s_branch .L0                              ; jump to label .L0
```

Translation: "Read my kernel arguments (pointers and sizes) from the
constant buffer at `s[0:1]`, wait for them to arrive, then jump past the
NOP-padded function prologue to the real code."

The `s_nop 0` block at lines 13-68 is **mandatory hardware-required NOP
padding** (gfx950 has specific alignment rules for code following a
branch). The README confirms: "NOP removal: 0% — All are hardware-required
hazard NOPs."

---

## 5. Workflow when reviewing an agent's diff

When an agent proposes a change, your review checklist is:

1. **What instruction did it move/add/remove?** Look it up in the CDNA4 ISA
   PDF (search for the mnemonic). Confirm the agent's claim about its
   semantics.
2. **Are the registers it touches dead?** If the agent claims `v[216:223]`
   is dead at the new location, search for `v216` through `v223` in the
   `.s` file and confirm those VGPRs aren't live across the change.
3. **Does the waitcnt ordering still hold?** If the agent moves a
   `buffer_load`, every `s_waitcnt vmcnt(N)` between old and new positions
   has its FIFO meaning changed. The agent should explicitly verify this.
4. **What's the bench number?** Run `co_compare --benchmark --warmup 50
   --iters 200`. Demand `cos >= 0.999999` for correctness. Demand a
   measurable speedup (>0.3%) that holds across 5 reruns.
5. **What's the cost?** Does the change extend a VGPR's live range and
   risk hurting occupancy? Did it add any new instructions? (Sometimes
   adding 4 instructions to save 200 cycles of stall is a win; sometimes
   it tips occupancy from 1 to 0.5 and is a 10% regression.)

If the agent can't answer (1)-(3) in its own words, with citations, it's
guessing. Don't merge.
